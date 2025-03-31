"""Unified MCP manager implementation."""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .mcp_types import ServerConfig, MCPConnection, ToolDefinition
from .mcp_errors import ErrorManager, MCPError

logger = logging.getLogger("mcplex")

# Maximum number of retries for operations
MAX_RETRIES = 3
# Delay between retries in seconds
RETRY_DELAY = 1

class MCPManager:
    """Unified manager for MCP operations."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._connections: Dict[str, MCPConnection] = {}
        self._tools: Dict[str, List[ToolDefinition]] = {}
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._response_futures: Dict[str, Dict[int, asyncio.Future]] = {}
        self._request_id = 0
        self._lock = asyncio.Lock()

    async def initialize_server(self, config: ServerConfig) -> bool:
        """Initialize a single MCP server connection."""
        try:
            if config.name in self._connections:
                return True

            # Expand path arguments
            expanded_args = [
                os.path.expanduser(a) if isinstance(a, str) and "~" in a else a
                for a in config.args
            ]

            # Prepare environment
            env_vars = os.environ.copy()
            if config.env:
                env_vars.update(config.env)

            # Start the process
            logger.info(f"Starting {config.name} with command: {config.command} {' '.join(expanded_args)}")
            
            process = await asyncio.create_subprocess_exec(
                config.command,
                *expanded_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars
            )

            connection = MCPConnection(config=config, process=process)
            self._connections[config.name] = connection
            self._message_queues[config.name] = asyncio.Queue()
            self._response_futures[config.name] = {}

            # Start background tasks
            asyncio.create_task(self._process_messages(config.name))
            asyncio.create_task(self._receive_messages(config.name))
            asyncio.create_task(self._monitor_stderr(config.name))

            # Initialize the connection
            if not await self._initialize_connection(config.name):
                await self._cleanup_server(config.name)
                return False

            # Fetch tools
            if not await self._fetch_tools(config.name):
                await self._cleanup_server(config.name)
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to initialize server {config.name}: {str(e)}")
            await self._cleanup_server(config.name)
            return False

    async def _initialize_connection(self, server_name: str) -> bool:
        """Initialize connection with protocol handshake."""
        try:
            connection = self._connections[server_name]
            self._request_id += 1
            req_id = self._request_id

            init_request = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {
                        "name": "MCPManager",
                        "version": "2.0.0"
                    }
                }
            }

            result = await self._send_request(
                server_name, 
                init_request,
                timeout=connection.config.timeout * connection.config.init_timeout_multiplier
            )

            if not result:
                return False

            # Send initialized notification
            await self._send_notification(server_name, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })

            return True

        except Exception as e:
            logger.error(f"Failed to initialize connection to {server_name}: {str(e)}")
            return False

    async def _fetch_tools(self, server_name: str) -> bool:
        """Fetch available tools from the server."""
        try:
            self._request_id += 1
            req_id = self._request_id

            result = await self._send_request(
                server_name,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/list",
                    "params": {}
                }
            )

            if not result or "tools" not in result:
                return False

            tools = [
                ToolDefinition(
                    name=t["name"],
                    server_name=server_name,
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema") or {"type": "object", "properties": {}}
                )
                for t in result["tools"]
            ]

            self._tools[server_name] = tools
            self._connections[server_name].tools = tools
            return True

        except Exception as e:
            logger.error(f"Failed to fetch tools from {server_name}: {str(e)}")
            return False

    async def _send_request(self, server_name: str, request: Dict, timeout: Optional[float] = None) -> Optional[Dict]:
        """Send a request and wait for response."""
        if not timeout:
            timeout = self._connections[server_name].config.timeout

        future = asyncio.Future()
        self._response_futures[server_name][request["id"]] = future
        
        await self._message_queues[server_name].put(request)
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            error = ErrorManager.handle_timeout(server_name, request.get("method", "unknown"), timeout)
            logger.error(str(error))
            return None
        finally:
            self._response_futures[server_name].pop(request["id"], None)

    async def _send_notification(self, server_name: str, notification: Dict) -> None:
        """Send a notification to the server."""
        await self._message_queues[server_name].put(notification)

    async def _process_messages(self, server_name: str) -> None:
        """Process outgoing messages for a server with retry logic."""
        retry_count = 0
        
        while True:
            try:
                if server_name not in self._connections:
                    break

                message = await self._message_queues[server_name].get()
                connection = self._connections[server_name]
                
                if not connection.process or connection.process.stdin.is_closing():
                    if retry_count < MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Connection issue with {server_name}, attempt {retry_count}/{MAX_RETRIES}")
                        await asyncio.sleep(RETRY_DELAY)
                        # Try to reconnect
                        if await self._reconnect_server(server_name):
                            connection = self._connections[server_name]
                            continue
                    break

                try:
                    data = json.dumps(message) + "\n"
                    connection.process.stdin.write(data.encode())
                    await connection.process.stdin.drain()
                    retry_count = 0  # Reset retry count on successful send
                except Exception as e:
                    logger.error(f"Error sending message to {server_name}: {str(e)}")
                    if retry_count < MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Retrying message send, attempt {retry_count}/{MAX_RETRIES}")
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processing for {server_name}: {str(e)}")
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.warning(f"Retrying after error, attempt {retry_count}/{MAX_RETRIES}")
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                break

    async def _receive_messages(self, server_name: str) -> None:
        """Process incoming messages from a server."""
        connection = self._connections[server_name]
        buffer = ""
        chunk_size = 8192  # Start with 8KB chunks
        
        while True:
            try:
                if not connection.process or connection.process.stdout.at_eof():
                    break

                chunk = await connection.process.stdout.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk.decode()
                
                # Process complete messages from buffer
                while '\n' in buffer:
                    try:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue
                            
                        message = json.loads(line.strip())
                        await self._handle_message(server_name, message)
                    except json.JSONDecodeError as je:
                        if "Unterminated string" in str(je) and len(buffer) < 1048576:  # 1MB limit
                            # Incomplete JSON, wait for more data
                            buffer = line + '\n' + buffer
                            break
                        logger.error(f"Invalid JSON from {server_name}: {line.strip()}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        continue

                # If buffer is getting too large, increase chunk size
                if len(buffer) > chunk_size * 2:
                    chunk_size = min(chunk_size * 2, 1048576)  # Max 1MB chunks

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving messages from {server_name}: {str(e)}")
                if "Separator is not found" in str(e):
                    # Clear buffer and continue if we hit message size limits
                    buffer = ""
                    continue
                break

    async def _handle_message(self, server_name: str, message: Dict) -> None:
        """Handle an incoming message from the server."""
        try:
            if "id" not in message:
                # Handle notifications
                if "method" in message:
                    logger.debug(f"Server {server_name} notification: {message['method']}")
                return

            if message["id"] not in self._response_futures[server_name]:
                logger.debug(f"No future found for message ID {message['id']} from {server_name}")
                return

            future = self._response_futures[server_name][message["id"]]
            
            if "result" in message:
                try:
                    future.set_result(message["result"])
                except asyncio.InvalidStateError:
                    logger.debug(f"Future already done for message ID {message['id']} from {server_name}")
            elif "error" in message:
                error = message["error"]
                if isinstance(error, dict):
                    error_msg = error.get("message", str(error))
                    error_code = error.get("code", 0)
                    error_data = error.get("data")
                    if error_data:
                        error_msg = f"{error_msg} - {error_data}"
                else:
                    error_msg = str(error)
                    error_code = 0
                
                future.set_exception(
                    ErrorManager.handle_protocol_error(
                        server_name,
                        f"Error {error_code}: {error_msg}"
                    )
                )
        except Exception as e:
            logger.error(f"Error handling message from {server_name}: {str(e)}")

    async def _monitor_stderr(self, server_name: str) -> None:
        """Monitor server's stderr output."""
        connection = self._connections[server_name]
        
        while True:
            try:
                if not connection.process or connection.process.stderr.at_eof():
                    break

                line = await connection.process.stderr.readline()
                if not line:
                    break

                stderr_line = line.decode().strip()
                if stderr_line:
                    logger.info(f"Server {server_name} stderr: {stderr_line}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring stderr for {server_name}: {str(e)}")
                break

    async def check_connection(self, server_name: str) -> bool:
        """Check if a server connection is healthy."""
        if server_name not in self._connections:
            return False
        connection = self._connections[server_name]
        return (connection.process and
                not connection.process.stdout.at_eof() and
                not connection.process.stdin.is_closing())

    async def _reconnect_server(self, server_name: str) -> bool:
        """Attempt to reconnect to a server."""
        try:
            if server_name not in self._connections:
                return False
            
            connection = self._connections[server_name]
            await self._cleanup_server(server_name)
            
            # Reinitialize with the same config
            return await self.initialize_server(connection.config)
            
        except Exception as e:
            logger.error(f"Failed to reconnect to {server_name}: {str(e)}")
            return False

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """Call a tool on a specific server with connection checks and retries."""
        retry_count = 0
        
        while True:
            try:
                if not await self.check_connection(server_name):
                    if retry_count < MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Server {server_name} connection unhealthy, attempting reconnect {retry_count}/{MAX_RETRIES}")
                        if await self._reconnect_server(server_name):
                            logger.info(f"Successfully reconnected to {server_name}")
                            continue
                        await asyncio.sleep(RETRY_DELAY)
                    raise MCPError(f"Server {server_name} disconnected and reconnection failed")

                self._request_id += 1
                req_id = self._request_id

                # Send request with timeout based on tool complexity
                result = await self._send_request(
                    server_name,
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    },
                    timeout=self._connections[server_name].config.timeout * 2
                )

                if not result:
                    if retry_count < MAX_RETRIES:
                        retry_count += 1
                        logger.warning(f"Tool call failed, retrying {retry_count}/{MAX_RETRIES}")
                        # Exponential backoff
                        await asyncio.sleep(RETRY_DELAY * (2 ** (retry_count - 1)))
                        continue
                    raise MCPError(f"Tool call failed after {MAX_RETRIES} retries: {tool_name}")

                # Validate response
                if isinstance(result, dict) and "error" in result:
                    raise MCPError(f"Tool returned error: {result['error']}")

                return result

            except asyncio.TimeoutError:
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.warning(f"Tool call timed out, retrying {retry_count}/{MAX_RETRIES}")
                    await asyncio.sleep(RETRY_DELAY * (2 ** (retry_count - 1)))
                    continue
                raise MCPError(f"Tool call timed out after {MAX_RETRIES} retries: {tool_name}")
            except Exception as e:
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.warning(f"Error in tool call, retrying {retry_count}/{MAX_RETRIES}: {str(e)}")
                    await asyncio.sleep(RETRY_DELAY * (2 ** (retry_count - 1)))
                    continue
                raise MCPError(f"Tool call failed: {str(e)}")

    async def _cleanup_server(self, server_name: str) -> None:
        """Clean up resources for a server."""
        if server_name not in self._connections:
            return

        connection = self._connections[server_name]
        if connection.process:
            try:
                connection.process.terminate()
                try:
                    await asyncio.wait_for(connection.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    connection.process.kill()
                    await connection.process.wait()
            except Exception as e:
                logger.error(f"Error cleaning up server {server_name}: {str(e)}")
            finally:
                if connection.process.stdin:
                    connection.process.stdin.close()

        self._connections.pop(server_name, None)
        self._tools.pop(server_name, None)
        self._message_queues.pop(server_name, None)
        self._response_futures.pop(server_name, None)

    async def cleanup(self) -> None:
        """Clean up all resources."""
        tasks = [self._cleanup_server(name) for name in list(self._connections.keys())]
        if tasks:
            await asyncio.gather(*tasks)

    @property
    def all_tools(self) -> List[ToolDefinition]:
        """Get all available tools across all servers."""
        return [
            tool for tools in self._tools.values()
            for tool in tools
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()