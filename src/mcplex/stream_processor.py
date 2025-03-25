"""
Optimized streaming implementation for MCPlex MCP.
Handles efficient streaming of responses and parallel tool call processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from .connection_pool import MCPConnectionPool

logger = logging.getLogger("mcplex")

class StreamProcessor:
    """
    Optimized streaming implementation with improved error handling and caching.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, connection_pool: MCPConnectionPool, quiet_mode: bool = False):
        self.connection_pool = connection_pool
        self.quiet_mode = quiet_mode
        self._server_groups_cache: Dict[str, str] = {}
        self._chunks_buffer = []

    def _get_server_name(self, func_name: str) -> str:
        """Get cached server name from function name."""
        if func_name not in self._server_groups_cache:
            self._server_groups_cache[func_name] = func_name.split("_", 1)[0]
        return self._server_groups_cache[func_name]

    async def process_tool_calls(self, tool_calls: List[Dict], servers_cfg: Dict) -> List[Dict]:
        """Process tool calls with improved error handling and caching."""
        server_groups: Dict[str, List[Dict]] = {}
        
        # Group tool calls using cached server names
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            srv_name = self._get_server_name(func_name)
            server_groups.setdefault(srv_name, []).append(tc)
            
            if not self.quiet_mode:
                print(f"\nProcessing tool call: {func_name}")
        
        async def process_server_group(srv_name: str, calls: List[Dict]) -> List[Dict]:
            results = []
            retries = 0
            
            while retries < self.MAX_RETRIES:
                try:
                    client = await self.connection_pool.get_connection(srv_name, servers_cfg[srv_name])
                    if not client:
                        raise ConnectionError(f"Could not get connection for server: {srv_name}")
                    
                    try:
                        for tc in calls:
                            result = await self._process_single_tool_call(client, tc)
                            results.append(result)
                        return results
                    finally:
                        await self.connection_pool.release_connection(srv_name, client)
                        
                except Exception as e:
                    retries += 1
                    if retries >= self.MAX_RETRIES:
                        logger.error(f"Failed to process tool calls for {srv_name} after {retries} retries: {str(e)}")
                        return [self._create_error_response(tc, {"error": str(e), "retries": retries}) for tc in calls]
                    logger.warning(f"Retry {retries} for {srv_name}: {str(e)}")
                    await asyncio.sleep(self.RETRY_DELAY * retries)
            
            return results

        # Process server groups in parallel with structured error handling
        tasks = [
            process_server_group(srv_name, calls)
            for srv_name, calls in server_groups.items()
        ]
        
        try:
            all_results = await asyncio.gather(*tasks)
            return [r for group in all_results for r in group]
        except Exception as e:
            logger.error(f"Failed to process tool calls: {str(e)}")
            return [self._create_error_response(tc, {"error": str(e)}) for tc in tool_calls]

    async def _process_single_tool_call(self, client: Any, tc: Dict) -> Dict:
        """Process a single tool call with improved error handling."""
        func_name = tc["function"]["name"]
        tool_name = func_name.split("_", 1)[1]
        
        try:
            func_args = json.loads(tc["function"].get("arguments", "{}"))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON arguments for {func_name}: {str(e)}")
            func_args = {}
        
        if not self.quiet_mode:
            print(f"Arguments: {json.dumps(func_args)}")
            
        result = await client.call_tool(tool_name, func_args)
        
        # if not self.quiet_mode:
        #     print(f"Result: {json.dumps(result.get('content', ''))}")
            
        
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": func_name,
            "content": json.dumps(result)
        }

    def _create_error_response(self, tc: Dict, error: Dict) -> Dict:
        """Create a structured error response."""
        error_content = {
            "status": "error",
            "details": error,
            "timestamp": asyncio.get_event_loop().time()
        }
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["function"]["name"],
            "content": json.dumps(error_content)
        }

    async def process_stream(self, generator: AsyncGenerator, conversation: List[Dict]) -> AsyncGenerator[str, None]:
        """Process stream with immediate token yielding."""
        try:
            current_tool_calls = []
            current_content = []
            
            async for chunk in generator:
                if chunk.get("is_chunk", False):
                    if chunk.get("token", False) and chunk.get("assistant_text"):
                        text = chunk["assistant_text"]
                        # Immediately yield each token
                        yield text
                        current_content.append(text)
                else:
                    # Handle tool calls and final text
                    tool_calls = chunk.get("tool_calls", [])
                    if tool_calls:
                        current_tool_calls.extend(tool_calls)
                    elif chunk.get("assistant_text"):
                        text = chunk["assistant_text"]
                        yield text
                        current_content.append(text)
            
            # Update conversation with complete message
            final_content = "".join(current_content)
            if final_content or current_tool_calls:
                conversation.append({
                    "role": "assistant",
                    "content": final_content,
                    "tool_calls": current_tool_calls
                })
                
        except Exception as e:
            error_msg = f"Stream processing error: {str(e)}"
            logger.error(error_msg)
            yield error_msg