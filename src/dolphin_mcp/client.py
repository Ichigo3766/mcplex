"""
Core client functionality for Dolphin MCP.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from .utils import load_mcp_config_from_file
from .providers.openai import generate_with_openai
from .providers.anthropic import generate_with_anthropic
from .providers.ollama import generate_with_ollama
from .connection_pool import MCPConnectionPool
from .stream_processor import StreamProcessor
from .mcp_client import MCPClient

# Global instances
_connection_pool = None
_all_functions = None

async def initialize_mcp(config: Optional[dict] = None, config_path: str = "mcp_config.json"):
    """Initialize MCP servers and cache their tool definitions.
    This should be called once when the application starts."""
    global _connection_pool, _all_functions
    
    if _connection_pool is not None and _all_functions:
        logger.info("MCP already initialized with %d tools", len(_all_functions))
        return  # Already initialized with tools
    
    logger.info("Starting MCP initialization...")
    
    # Load or use provided config
    if config is None:
        logger.info("Loading config from %s", config_path)
        config = await load_mcp_config_from_file(config_path)
    servers_cfg = config.get("mcpServers", {})
    logger.info("Found %d MCP servers in config", len(servers_cfg))
    
    # Initialize connection pool and tools list
    _connection_pool = MCPConnectionPool(max_connections=10)
    _all_functions = []
    
    # Get tool definitions from all servers
    logger.info("Connecting to MCP servers and retrieving tool definitions...")
    tasks = {}
    for server_name, conf in servers_cfg.items():
        client = await _connection_pool.get_connection(server_name, conf)
        if client:
            tasks[server_name] = client.list_tools()
    
    if tasks:
        logger.info("Found %d servers to connect to", len(tasks))
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for server_name, tools in zip(tasks.keys(), results):
            logger.info("Processing tools for server %s", server_name)
            if isinstance(tools, Exception):
                logger.error(f"Error getting tools for {server_name}: {str(tools)}")
                continue
            for t in tools:
                input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                fn_def = {
                    "name": f"{server_name}_{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": input_schema
                }
                _all_functions.append(fn_def)
    
    logger.info("MCP initialization complete. Registered %d tools", len(_all_functions))
    if not _all_functions:
        raise RuntimeError("No MCP servers could be started.")


async def shutdown():
    """Cleanup the global connection pool when application shuts down."""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.cleanup()
        _connection_pool = None

logger = logging.getLogger("dolphin_mcp")

async def generate_text(conversation: List[Dict], model_cfg: Dict,
                       all_functions: List[Dict], stream: bool = False) -> Union[Dict, AsyncGenerator]:
    """
    Generate text using the specified provider.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        stream: Whether to stream the response
        
    Returns:
        If stream=False: Dict containing assistant_text and tool_calls
        If stream=True: AsyncGenerator yielding chunks of assistant text and tool calls
    """
    provider = model_cfg.get("provider", "").lower()
    
    if provider == "openai":
        if stream:
            return await generate_with_openai(conversation, model_cfg, all_functions, stream=True)
        else:
            return await generate_with_openai(conversation, model_cfg, all_functions, stream=False)
    
    # For non-streaming providers, wrap the response in an async generator if streaming is requested
    if stream:
        async def wrap_response():
            if provider == "anthropic":
                result = await generate_with_anthropic(conversation, model_cfg, all_functions)
            elif provider == "ollama":
                result = await generate_with_ollama(conversation, model_cfg, all_functions)
            else:
                result = {"assistant_text": f"Unsupported provider '{provider}'", "tool_calls": []}
            yield result
        return wrap_response()
    
    # Non-streaming path
    if provider == "anthropic":
        return await generate_with_anthropic(conversation, model_cfg, all_functions)
    elif provider == "ollama":
        return await generate_with_ollama(conversation, model_cfg, all_functions)
    else:
        return {"assistant_text": f"Unsupported provider '{provider}'", "tool_calls": []}

async def log_messages_to_file(messages: List[Dict], functions: List[Dict], log_path: str):
    """
    Log messages and function definitions to a JSONL file.
    
    Args:
        messages: List of messages to log
        functions: List of function definitions
        log_path: Path to the log file
    """
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Append to file
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "messages": messages,
                "functions": functions
            }) + "\n")
    except Exception as e:
        logger.error(f"Error logging messages to {log_path}: {str(e)}")

async def run_interaction(
    user_query: str,
    model_name: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: str = "mcp_config.json",
    quiet_mode: bool = False,
    log_messages_path: Optional[str] = None,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Run an interaction with the MCP servers.
    
    Args:
        user_query: The user's query
        model_name: Name of the model to use (optional)
        config: Configuration dict (optional, if not provided will load from config_path)
        config_path: Path to the configuration file (default: mcp_config.json)
        quiet_mode: Whether to suppress intermediate output (default: False)
        log_messages_path: Path to log messages in JSONL format (optional)
        stream: Whether to stream the response (default: False)
        
    Returns:
        If stream=False: The final text response
        If stream=True: AsyncGenerator yielding chunks of the response
    """
    # 1) If config is not provided, load from file:
    if config is None:
        config = await load_mcp_config_from_file(config_path)

    servers_cfg = config.get("mcpServers", {})
    models_cfg = config.get("models", [])

    # 2) Choose a model
    chosen_model = None
    if model_name:
        for m in models_cfg:
            if m.get("model") == model_name:
                chosen_model = m
                break
        if not chosen_model:
            # fallback to default or fail
            for m in models_cfg:
                if m.get("default"):
                    chosen_model = m
                    break
    else:
        # if model_name not specified, pick default
        for m in models_cfg:
            if m.get("default"):
                chosen_model = m
                break
        if not chosen_model and models_cfg:
            chosen_model = models_cfg[0]

    if not chosen_model:
        error_msg = "No suitable model found in config."
        if stream:
            async def error_gen():
                yield error_msg
            return error_gen()
        return error_msg

    # 3) Get or create global connection pool and stream processor
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = MCPConnectionPool(max_connections=10)
    stream_processor = StreamProcessor(_connection_pool, quiet_mode)

    # 4) Initialize MCP if not already done
    if _connection_pool is None or not _all_functions:  # Check if None or empty
        await initialize_mcp(config=config, config_path=config_path)
        if not _all_functions:  # If still empty after initialization
            error_msg = "Failed to initialize MCP servers. No tools were registered."
            if stream:
                async def error_gen():
                    yield error_msg
                return error_gen()
            return error_msg


    # 5) Build conversation
    system_msg = chosen_model.get("systemMessage", "You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query}
    ]

    async def cleanup():
        """Log messages if needed"""
        if log_messages_path:
            await log_messages_to_file(conversation, _all_functions, log_messages_path)

    if stream:
        async def stream_response():
            try:
                while True:  # Main conversation loop
                    generator = await generate_text(conversation, chosen_model, _all_functions, stream=True)
                    async for chunk in  stream_processor.process_stream(generator, conversation):
                        yield chunk
                    
                    # Break if no more tool calls to process
                    if not conversation[-1].get("tool_calls"):
                        break
                    
                    # Process tool calls and add results to conversation
                    tool_calls = conversation[-1].get("tool_calls", [])
                    results = await stream_processor.process_tool_calls(tool_calls, servers_cfg)
                    conversation.extend(results)
                    
            finally:
                await cleanup()
        
        return stream_response()
    else:
        try:
            final_text = ""
            while True:
                gen_result = await generate_text(conversation, chosen_model, _all_functions, stream=False)
                
                assistant_text = gen_result["assistant_text"]
                final_text = assistant_text
                tool_calls = gen_result.get("tool_calls", [])

                # Add the assistant's message
                assistant_message = {"role": "assistant", "content": assistant_text}
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                conversation.append(assistant_message)
                logger.info(f"Added assistant message: {json.dumps(assistant_message, indent=2)}")

                if not tool_calls:
                    break

                results = await stream_processor.process_tool_calls(tool_calls, servers_cfg)
                conversation.extend(results)

            return final_text
        finally:
            await cleanup()
