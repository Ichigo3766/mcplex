"""
Optimized streaming implementation for Dolphin MCP.
Handles efficient streaming of responses and parallel tool call processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from .connection_pool import MCPConnectionPool

logger = logging.getLogger("dolphin_mcp")

class StreamProcessor:
    """
    Handles optimized streaming of responses and tool call processing.
    
    Features:
    - Efficient token streaming without unnecessary accumulation
    - Parallel tool call processing
    - Server-grouped tool call execution
    - Connection pooling integration
    """
    
    def __init__(self, connection_pool: MCPConnectionPool, quiet_mode: bool = False):
        """
        Initialize the stream processor.
        
        Args:
            connection_pool: Connection pool for MCP servers
            quiet_mode: Whether to suppress intermediate output
        """
        self.connection_pool = connection_pool
        self.quiet_mode = quiet_mode

    async def process_tool_calls(self, tool_calls: List[Dict], servers_cfg: Dict) -> List[Dict]:
        """
        Process multiple tool calls efficiently by grouping by server.
        
        Args:
            tool_calls: List of tool calls to process
            servers_cfg: Server configuration dictionary
            
        Returns:
            List of tool call results
        """
        # Group tool calls by server
        server_groups: Dict[str, List[Dict]] = {}
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            srv_name = func_name.split("_", 1)[0]
            
            if srv_name not in server_groups:
                server_groups[srv_name] = []
            server_groups[srv_name].append(tc)
            
            print(f"\n[{srv_name}] Processing tool call: {func_name}")
            
        # Process each server's tool calls in parallel
        async def process_server_group(srv_name: str, calls: List[Dict]) -> List[Dict]:
            results = []
            client = await self.connection_pool.get_connection(srv_name, servers_cfg[srv_name])
            
            if not client:
                error_result = {
                    "error": f"Could not get connection for server: {srv_name}"
                }
                return [self._create_error_response(tc, error_result) for tc in calls]
                
            try:
                for tc in calls:
                    result = await self._process_single_tool_call(client, tc)
                    results.append(result)
            finally:
                await self.connection_pool.release_connection(srv_name, client)
                
            return results
            
        # Create tasks for each server group
        tasks = [
            process_server_group(srv_name, calls)
            for srv_name, calls in server_groups.items()
        ]
        
        # Execute all groups in parallel
        all_results = await asyncio.gather(*tasks)
        return [r for group in all_results for r in group]  # Flatten results

    async def _process_single_tool_call(self, client: Any, tc: Dict) -> Dict:
        """Process a single tool call using the provided client."""
        func_name = tc["function"]["name"]
        tool_name = func_name.split("_", 1)[1]
        
        try:
            func_args = json.loads(tc["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            func_args = {}
            
        if not self.quiet_mode:
            print(f"  Arguments: {json.dumps(func_args)}")
            
        result = await client.call_tool(tool_name, func_args)
        
        if not self.quiet_mode:
            print(f"  Result: {json.dumps(result, indent=2)}")
            
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": func_name,
            "content": json.dumps(result)
        }

    def _create_error_response(self, tc: Dict, error: Dict) -> Dict:
        """Create an error response for a failed tool call."""
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["function"]["name"],
            "content": json.dumps(error)
        }

    async def process_stream(self, generator: AsyncGenerator, conversation: List[Dict]) -> AsyncGenerator[str, None]:
        """
        Process a response stream efficiently.
        
        Args:
            generator: The response generator
            conversation: The conversation history to update
            
        Yields:
            Response tokens and processes tool calls
        """
        try:
            current_tool_calls = []
            current_content = ""
            last_chunk_was_tool_call = False
            
            async for chunk in generator:
                if chunk.get("is_chunk", False):
                    if chunk.get("token", False) and chunk.get("assistant_text"):
                        yield chunk["assistant_text"]
                        current_content += chunk["assistant_text"]
                else:
                    # Handle tool calls from the final chunk
                    tool_calls = chunk.get("tool_calls", [])
                    if tool_calls:
                        current_tool_calls.extend(tool_calls)
                        last_chunk_was_tool_call = True
                    # Only yield final text if this wasn't already streamed
                    elif chunk.get("assistant_text") and not last_chunk_was_tool_call:
                        yield chunk["assistant_text"]
                        current_content += chunk["assistant_text"]
            
            # Update conversation with final message
            if current_content or current_tool_calls:
                conversation.append({
                    "role": "assistant",
                    "content": current_content,
                    "tool_calls": current_tool_calls
                })
                
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            yield f"Error processing stream: {str(e)}"