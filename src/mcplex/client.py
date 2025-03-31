"""
Main MCP client interface.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator

from .mcp_types import ServerConfig
from .mcp_manager import MCPManager
from .mcp_errors import MCPError, ConfigurationError
from .utils import load_mcp_config_from_file
from .providers.openai import generate_with_openai
from .providers.anthropic import generate_with_anthropic
from .providers.ollama import generate_with_ollama

logger = logging.getLogger("mcplex")

# Maximum retries for tool calls at client level
CLIENT_MAX_RETRIES = 3
# Delay between retries in seconds
CLIENT_RETRY_DELAY = 1

class MCPClient:
    """Main interface for MCP operations."""
    
    def __init__(self):
        self.manager = MCPManager()
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self, config: Optional[Dict] = None, config_path: str = "mcp_config.json", quiet_mode: bool = False) -> bool:
        """Initialize MCP with configuration."""
        async with self._lock:
            if self._initialized:
                return True

            try:
                if config is None:
                    config = await load_mcp_config_from_file(config_path)

                servers_cfg = {
                    name: conf for name, conf in config.get("mcpServers", {}).items()
                    if not conf.get("disabled", False)
                }

                if not servers_cfg:
                    raise ConfigurationError("No enabled MCP servers found in configuration")

                # Initialize each server
                for name, conf in servers_cfg.items():
                    if not quiet_mode:
                        logger.info(f"Initializing server: {name}")
                    
                    server_config = ServerConfig(
                        name=name,
                        command=conf["command"],
                        args=conf.get("args", []),
                        env=conf.get("env"),
                        timeout=conf.get("timeout", 30),
                        init_timeout_multiplier=conf.get("init_timeout_multiplier", 1)
                    )

                    if not await self.manager.initialize_server(server_config):
                        logger.error(f"Failed to initialize server: {name}")
                        continue

                    if not quiet_mode:
                        logger.info(f"Successfully initialized server: {name}")

                self._initialized = bool(self.manager.all_tools)
                return self._initialized

            except Exception as e:
                logger.error(f"Initialization failed: {str(e)}")
                return False

    async def shutdown(self):
        """Clean up resources."""
        await self.manager.cleanup()
        self._initialized = False

    def _select_model(self, models_cfg: List[Dict], model_name: Optional[str] = None) -> Optional[Dict]:
        """Select appropriate model configuration."""
        if not models_cfg:
            return None
            
        if model_name:
            model_name_lower = model_name.lower()
            model = next(
                (m for m in models_cfg 
                 if m.get("model", "").lower() == model_name_lower 
                 or m.get("title", "").lower() == model_name_lower),
                next((m for m in models_cfg if m.get("default")), models_cfg[0])
            )
        else:
            model = next((m for m in models_cfg if m.get("default")), models_cfg[0])
        
        # Handle systemMessagePath if present
        if model and "systemMessagePath" in model:
            try:
                # Get the directory of the config file to make paths relative to it
                config_dir = os.path.dirname(os.path.abspath("mcp_config.json"))
                system_message_path = os.path.join(config_dir, model["systemMessagePath"])
                
                with open(system_message_path, "r") as f:
                    model["systemMessage"] = f.read().strip()
            except Exception as e:
                logger.error(f"Error loading system message from {model.get('systemMessagePath')}: {str(e)}")
                # Fall back to default system message if file loading fails
                if "systemMessage" not in model:
                    model["systemMessage"] = "You are a helpful assistant with access to MCP servers. You will carefully examine the query and use MCP servers IF needed to answer the query."
        
        return model

    async def generate_text(self, conversation: List[Dict], model_cfg: Dict, stream: bool = False) -> Union[Dict, AsyncGenerator]:
        """Generate text using the specified provider."""
        provider = model_cfg.get("provider", "").lower()
        
        provider_map = {
            "openai": generate_with_openai,
            "anthropic": generate_with_anthropic,
            "ollama": generate_with_ollama
        }
        
        if provider not in provider_map:
            error_result = {"assistant_text": f"Unsupported provider '{provider}'", "tool_calls": []}
            if stream:
                async def error_stream():
                    yield error_result
                return error_stream()
            return error_result
        
        provider_func = provider_map[provider]
        
        # Convert ToolDefinition objects to dictionaries
        tools = [tool.to_dict() for tool in self.manager.all_tools]
        
        if stream:
            if provider == "openai":
                return await provider_func(conversation, model_cfg, tools, stream=True)
            else:
                async def wrap_stream():
                    result = await provider_func(conversation, model_cfg, tools)
                    yield result
                return wrap_stream()
        else:
            return await provider_func(conversation, model_cfg, tools, stream=False)

    async def process_tool_calls(self, tool_calls: List[Dict], servers_cfg: Dict) -> List[Dict]:
        """Process tool calls with retry logic and return results."""
        results = []
        
        for call in tool_calls:
            retry_count = 0
            while True:
                try:
                    # Handle both function_call and tool_calls format
                    if "function" in call:
                        # OpenAI format
                        name = call["function"]["name"]
                        arguments = json.loads(call["function"].get("arguments", "{}"))
                        call_id = call.get("id", "")
                    elif "name" in call:
                        # Direct format
                        name = call["name"]
                        arguments = call.get("arguments", {})
                        call_id = call.get("id", "")
                    else:
                        raise MCPError("Invalid tool call format")

                    name_parts = name.split("_", 1)
                    if len(name_parts) != 2:
                        raise MCPError(f"Invalid tool name format: {name}")
                    
                    server_name, tool_name = name_parts
                    if server_name not in servers_cfg:
                        raise MCPError(f"Unknown server: {server_name}")

                    # Check server connection status
                    if not await self.manager.check_connection(server_name):
                        if retry_count < CLIENT_MAX_RETRIES:
                            retry_count += 1
                            logger.warning(f"Server {server_name} connection issue, retrying {retry_count}/{CLIENT_MAX_RETRIES}")
                            await asyncio.sleep(CLIENT_RETRY_DELAY)
                            continue
                        raise MCPError(f"Server {server_name} connection failed after {CLIENT_MAX_RETRIES} retries")

                    result = await self.manager.call_tool(
                        server_name,
                        tool_name,
                        arguments
                    )

                    results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result)
                    })
                    break  # Success, exit retry loop

                except Exception as e:
                    error_msg = f"Tool call failed: {str(e)}"
                    logger.error(error_msg)
                    
                    if retry_count < CLIENT_MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Retrying tool call {retry_count}/{CLIENT_MAX_RETRIES}")
                        await asyncio.sleep(CLIENT_RETRY_DELAY)
                        continue
                        
                    # After all retries failed, add error result
                    if "function" in call:
                        name = call["function"].get("name", "unknown")
                        call_id = call.get("id", "")
                    else:
                        name = call.get("name", "unknown")
                        call_id = call.get("id", "")
                        
                    results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps({
                            "error": f"{error_msg} (after {CLIENT_MAX_RETRIES} retries)"
                        })
                    })
                    break  # Exit retry loop after max retries

        return results

    async def run_interaction(
        self,
        user_query: str,
        model_name: Optional[str] = None,
        config: Optional[dict] = None,
        config_path: str = "mcp_config.json",
        quiet_mode: bool = False,
        log_messages_path: Optional[str] = None,
        stream: bool = False,
        show_tool_calls: bool = False
    ) -> Union[str, AsyncGenerator[Union[str, Dict], None]]:
        """Run an interaction with the MCP system."""
        
        if config is None:
            config = await load_mcp_config_from_file(config_path)
        
        chosen_model = self._select_model(config.get("models", []), model_name)
        if not chosen_model:
            error_msg = "No suitable model found in config."
            if stream:
                async def error_stream():
                    yield error_msg
                return error_stream()
            return error_msg
        
        if not await self.initialize(config, config_path, quiet_mode):
            error_msg = "Failed to initialize MCP servers."
            if stream:
                async def error_stream():
                    yield error_msg
                return error_stream()
            return error_msg
        
        conversation = [
            {
                "role": "system",
                "content": chosen_model.get(
                    "systemMessage",
                    "You are a helpful assistant with access to MCP servers. You will carefully examine the query and use MCP servers IF needed to answer the query."
                )
            },
            {"role": "user", "content": user_query}
        ]
        
        async def log_messages():
            """Log messages if path provided."""
            if log_messages_path:
                try:
                    os.makedirs(os.path.dirname(log_messages_path), exist_ok=True)
                    with open(log_messages_path, "a") as f:
                        f.write(json.dumps({
                            "messages": conversation,
                            "functions": [t.to_dict() for t in self.manager.all_tools]
                        }) + "\n")
                except Exception as e:
                    logger.error(f"Error logging messages to {log_messages_path}: {str(e)}")
        
        if stream:
            async def stream_response():
                try:
                    needs_continuation = True
                    while needs_continuation:
                        generator = await self.generate_text(conversation, chosen_model, stream=True)
                        current_text = []
                        has_tool_calls = False
                        
                        async for chunk in generator:
                            if isinstance(chunk, dict):
                                # Extract text content from chunk
                                text = chunk.get("assistant_text", "")
                                is_chunk = chunk.get("is_chunk", False)
                                tool_calls = chunk.get("tool_calls", [])
                                
                                # Process text if present
                                if text:
                                    if is_chunk:
                                        current_text.append(text)
                                        yield text
                                    else:
                                        yield text
                                        current_text = [text]

                                # Process tool calls if present
                                if tool_calls:
                                    has_tool_calls = True
                                    # Wait for any remaining text chunks
                                    async for remaining in generator:
                                        if isinstance(remaining, dict) and remaining.get("assistant_text"):
                                            current_text.append(remaining["assistant_text"])
                                            yield remaining["assistant_text"]
                                    
                                    # Update conversation with complete text
                                    conversation.append({
                                        "role": "assistant",
                                        "content": "".join(current_text),
                                        "tool_calls": tool_calls
                                    })

                                    if show_tool_calls:
                                        for call in tool_calls:
                                            if "function" in call:
                                                name = call["function"]["name"]
                                                args = call["function"].get("arguments", "{}")
                                            else:
                                                name = call["name"]
                                                args = json.dumps(call.get("arguments", {}))
                                            yield f"\n[Tool Call] {name}({args})"
                                    
                                    # Process tool calls and wait for results
                                    results = await self.process_tool_calls(
                                        tool_calls,
                                        config.get("mcpServers", {})
                                    )
                                    
                                    # Add tool results to conversation
                                    conversation.extend(results)
                                    
                                    # Yield tool results with better formatting
                                    for result in results:
                                        content = json.loads(result["content"])
                                        if isinstance(content, dict) and "error" in content:
                                            yield f"\n[Error] {content['error']}"
                                        else:
                                            if show_tool_calls:
                                                yield f"\n[Result] {json.dumps(content, indent=2)}"
                                            else:
                                                yield f"\n{json.dumps(content)}"
                                    
                                    # Exit this generator to get assistant's response to tool results
                                    break
                            else:
                                # If chunk is a string (error message), yield it directly
                                yield chunk
                                if not is_chunk:
                                    break
                        
                        # Add final message to conversation if we have content and no tool calls
                        if current_text and not has_tool_calls:
                            conversation.append({
                                "role": "assistant",
                                "content": "".join(current_text)
                            })
                        
                        # Continue only if we had tool calls
                        needs_continuation = has_tool_calls
                        
                except Exception as e:
                    error_msg = f"\nError in stream response: {str(e)}"
                    logger.error(error_msg)
                    yield error_msg
                finally:
                    await log_messages()
                    
            return stream_response()
        
        try:
            final_text = []
            while True:
                try:
                    gen_result = await self.generate_text(conversation, chosen_model, stream=False)
                    
                    assistant_text = gen_result["assistant_text"]
                    tool_calls = gen_result.get("tool_calls", [])
                    
                    final_text.append(assistant_text)
                    
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_text,
                        **({"tool_calls": tool_calls} if tool_calls else {})
                    })
                    
                    if not tool_calls:
                        break

                    if show_tool_calls:
                        # Show tool calls being made
                        for call in tool_calls:
                            if "function" in call:
                                name = call["function"]["name"]
                                args = call["function"].get("arguments", "{}")
                            else:
                                name = call["name"]
                                args = json.dumps(call.get("arguments", {}))
                            final_text.append(f"\n[Tool Call] {name}({args})")
                    
                    results = await self.process_tool_calls(
                        tool_calls,
                        config.get("mcpServers", {})
                    )
                    conversation.extend(results)

                    # Add formatted results
                    for result in results:
                        content = json.loads(result["content"])
                        if isinstance(content, dict) and "error" in content:
                            final_text.append(f"\n[Error] {content['error']}")
                        else:
                            if show_tool_calls:
                                final_text.append(f"\n[Result] {json.dumps(content, indent=2)}")
                            else:
                                final_text.append(f"\n{json.dumps(content)}")
                    
                except Exception as e:
                    logger.error(f"Error in non-stream response: {str(e)}")
                    return f"Error: {str(e)}"
            
            return "".join(final_text)
        finally:
            await log_messages()

# Global client instance
_client = MCPClient()

async def initialize_mcp(config: Optional[dict] = None, config_path: str = "mcp_config.json", quiet_mode: bool = False) -> bool:
    """Initialize the global MCP client."""
    return await _client.initialize(config, config_path, quiet_mode)

async def shutdown():
    """Shutdown the global MCP client."""
    await _client.shutdown()

async def run_interaction(
    user_query: str,
    model_name: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: str = "mcp_config.json",
    quiet_mode: bool = False,
    log_messages_path: Optional[str] = None,
    stream: bool = False,
    show_tool_calls: bool = False
) -> Union[str, AsyncGenerator[Union[str, Dict], None]]:
    """
    Run an interaction using the global MCP client.
    
    Args:
        user_query: The user's input query
        model_name: Optional name of the model to use
        config: Optional configuration dictionary
        config_path: Path to config file if config not provided
        quiet_mode: Whether to suppress initialization logs
        log_messages_path: Optional path to log messages
        stream: Whether to stream the response
        show_tool_calls: Whether to show detailed tool calls and results
    """
    return await _client.run_interaction(
        user_query=user_query,
        model_name=model_name,
        config=config,
        config_path=config_path,
        quiet_mode=quiet_mode,
        log_messages_path=log_messages_path,
        stream=stream,
        show_tool_calls=show_tool_calls
    )
