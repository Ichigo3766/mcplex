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

logger = logging.getLogger("dolphin_mcp")

class MCPState:
    """Singleton class to manage MCP global state."""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPState, cls).__new__(cls)
            cls._instance.connection_pool = None
            cls._instance.stream_processor = None
            cls._instance.all_functions = []
            cls._instance._initializing = False
            cls._instance._server_connections = {}
            cls._instance._tool_cache = {}
        return cls._instance

    @property
    def initialized(self) -> bool:
        return bool(self._tool_cache)

    async def ensure_initialized(self, config: dict) -> bool:
        """Ensure MCP is initialized, with connection pooling and caching."""
        async with self._lock:
            if self.initialized:
                return True
                
            if self._initializing:
                while self._initializing:
                    await asyncio.sleep(0.1)
                return self.initialized

            self._initializing = True
            try:
                return await self._initialize(config)
            finally:
                self._initializing = False

    async def _initialize(self, config: dict) -> bool:
        """Internal initialization with efficient connection handling."""
        if not self.connection_pool:
            self.connection_pool = MCPConnectionPool(max_connections=10)
            self.stream_processor = StreamProcessor(self.connection_pool)
        
        servers_cfg = config.get("mcpServers", {})
        if not servers_cfg:
            logger.error("No MCP servers configured")
            return False

        # Initialize connections and fetch tools only for servers we haven't cached
        uncached_servers = {
            name: conf for name, conf in servers_cfg.items()
            if name not in self._tool_cache
        }

        if uncached_servers:
            tasks = {
                name: self.connection_pool.get_connection(name, conf)
                for name, conf in uncached_servers.items()
            }
            
            clients = await asyncio.gather(*tasks.values(), return_exceptions=True)
            active_clients = {
                name: client for name, client in zip(tasks.keys(), clients)
                if not isinstance(client, Exception) and client is not None
            }

            # Fetch and cache tools for new servers
            tool_tasks = [
                self._fetch_and_cache_tools(name, client)
                for name, client in active_clients.items()
            ]
            await asyncio.gather(*tool_tasks, return_exceptions=True)

        # Update all_functions from cache
        self.all_functions = [
            tool for tools in self._tool_cache.values()
            for tool in tools
        ]

        return bool(self.all_functions)

    async def _fetch_and_cache_tools(self, server_name: str, client: Any) -> None:
        """Fetch and cache tools for a server."""
        try:
            tools = await client.list_tools()
            self._tool_cache[server_name] = [
                {
                    "name": f"{server_name}_{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}
                } for t in tools
            ]
        except Exception as e:
            logger.error(f"Error fetching tools for {server_name}: {str(e)}")
            self._tool_cache[server_name] = []

    def reset(self):
        """Reset the state."""
        self.connection_pool = None
        self.all_functions = []
        self._tool_cache = {}
        self._server_connections = {}

_state = MCPState()

async def initialize_mcp(config: Optional[dict] = None, config_path: str = "mcp_config.json"):
    """Initialize MCP servers and cache their tool definitions."""
    if config is None:
        config = await load_mcp_config_from_file(config_path)
    
    return await _state.ensure_initialized(config)

async def shutdown():
    """Cleanup the global connection pool when application shuts down."""
    if _state.connection_pool:
        await _state.connection_pool.cleanup()
        _state.reset()

def _select_model(models_cfg: List[Dict], model_name: Optional[str] = None) -> Optional[Dict]:
    """Helper function to select appropriate model configuration."""
    if not models_cfg:
        return None
        
    if model_name:
        return next((m for m in models_cfg if m.get("model") == model_name or m.get("title") == model_name), 
                   next((m for m in models_cfg if m.get("default")), models_cfg[0]))
    
    return next((m for m in models_cfg if m.get("default")), models_cfg[0])

async def generate_text(conversation: List[Dict], model_cfg: Dict,
                       all_functions: List[Dict], stream: bool = False) -> Union[Dict, AsyncGenerator]:
    """Generate text using the specified provider."""
    provider = model_cfg.get("provider", "").lower()
    
    # Map providers to their generation functions
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
    
    if stream:
        if provider == "openai":
            return await provider_func(conversation, model_cfg, all_functions, stream=True)
        else:
            async def wrap_stream():
                result = await provider_func(conversation, model_cfg, all_functions)
                yield result
            return wrap_stream()
    else:
        return await provider_func(conversation, model_cfg, all_functions, stream=False)

async def log_messages_to_file(messages: List[Dict], functions: List[Dict], log_path: str):
    """Log messages and function definitions to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"messages": messages, "functions": functions}) + "\n")
    except Exception as e:
        logger.error(f"Error logging messages to {log_path}: {str(e)}")

async def run_interaction(
    user_query: str,
    model_name: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: str = "mcp_config.json",
    show_tool_calls: bool = True,
    log_messages_path: Optional[str] = None,
    stream: bool = False
) -> Union[str, AsyncGenerator[Union[str, Dict], None]]:
    """Run an interaction with the MCP servers."""
    
    # Load configuration if not provided
    if config is None:
        config = await load_mcp_config_from_file(config_path)
    
    # Select model
    chosen_model = _select_model(config.get("models", []), model_name)
    if not chosen_model:
        error_msg = "No suitable model found in config."
        if stream:
            async def error_stream():
                yield error_msg
            return error_stream()
        return error_msg
    
    # Ensure MCP is initialized (now using efficient async initialization)
    if not await _state.ensure_initialized(config):
        error_msg = "Failed to initialize MCP servers. No tools were registered."
        if stream:
            async def error_stream():
                yield error_msg
            return error_stream()
        return error_msg
    
    # Initialize conversation
    conversation = [
        {"role": "system", "content": chosen_model.get("systemMessage", "You are a helpful assistant with access to MCP servers. You will carefully examine the query and use MCP servers IF needed to answer the query.")},
        {"role": "user", "content": user_query}
    ]
    
    # Use singleton stream processor
    async def cleanup():
        """Log messages and cleanup if needed"""
        if log_messages_path:
            await log_messages_to_file(conversation, _state.all_functions, log_messages_path)
    
    if stream:
        async def stream_response():
            try:
                while True:
                    generator = await generate_text(conversation, chosen_model, _state.all_functions, stream=True)
                    async for chunk in _state.stream_processor.process_stream(generator, conversation):
                        yield chunk
                    
                    if not conversation[-1].get("tool_calls"):
                        break
                    
                    tool_calls = conversation[-1].get("tool_calls", [])
                    try:
                        results = await _state.stream_processor.process_tool_calls(tool_calls, config.get("mcpServers", {}))
                        conversation.extend(results)
                    except Exception as e:
                        logger.error(f"Error processing tool calls: {str(e)}")
                        yield f"\nError processing tool calls: {str(e)}"
                        break
            except Exception as e:
                logger.error(f"Error in stream response: {str(e)}")
                yield f"\nError: {str(e)}"
            finally:
                await cleanup()
        return stream_response()
    
    try:
        final_text = ""
        while True:
            try:
                gen_result = await generate_text(conversation, chosen_model, _state.all_functions, stream=False)
                
                assistant_text = gen_result["assistant_text"]
                final_text = assistant_text
                tool_calls = gen_result.get("tool_calls", [])
                
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_text,
                    **({"tool_calls": tool_calls} if tool_calls else {})
                }
                conversation.append(assistant_message)
                
                if not tool_calls:
                    break
                
                results = await _state.stream_processor.process_tool_calls(tool_calls, config.get("mcpServers", {}))
                conversation.extend(results)
            except Exception as e:
                logger.error(f"Error in non-stream response: {str(e)}")
                return f"Error: {str(e)}"
        
        return final_text
    finally:
        await cleanup()
