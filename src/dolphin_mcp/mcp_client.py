"""
MCP client class implementation.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("dolphin_mcp")

class MCPClient:
    """Implementation for a single MCP server."""
    def __init__(self, server_name: str, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, timeout: float = 3600.0):
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout  # Single timeout for MCP server responses
        
        # Internal state
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: List[Dict[str, Any]] = []
        self.request_id: int = 0
        self.protocol_version: str = "2024-11-05"
        self.receive_task: Optional[asyncio.Task] = None
        self.responses: Dict[int, Dict] = {}
        self.server_capabilities: Dict[str, Any] = {}
        self._shutdown: bool = False
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._send_lock: asyncio.Lock = asyncio.Lock()

    async def _process_queue(self):
        """Process messages from the queue to avoid concurrent writes."""
        while not self._shutdown:
            try:
                message = await self._message_queue.get()
                async with self._send_lock:
                    if not self.process or self._shutdown:
                        continue
                    try:
                        data = json.dumps(message) + "\n"
                        self.process.stdin.write(data.encode())
                        await self.process.stdin.drain()
                    except Exception as e:
                        logger.error(f"Server {self.server_name}: Error sending message: {str(e)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Server {self.server_name}: Queue processing error: {str(e)}")
            finally:
                self._message_queue.task_done()

    async def _receive_loop(self):
        """Handle incoming messages from the server."""
        if not self.process or self.process.stdout.at_eof():
            return
        
        try:
            while not self.process.stdout.at_eof() and not self._shutdown:
                try:
                    line = await self.process.stdout.readline()
                    if not line:
                        break
                    
                    message = json.loads(line.decode().strip())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON
                except Exception as e:
                    logger.error(f"Server {self.server_name}: Error processing message: {str(e)}")
        except asyncio.CancelledError:
            pass
        finally:
            # Ensure cleanup happens if receive loop exits
            if not self._shutdown:
                asyncio.create_task(self.stop())

    async def _handle_message(self, message: dict):
        """Process incoming messages based on type."""
        if "jsonrpc" not in message:
            return
        
        if "id" in message:
            if "result" in message or "error" in message:
                self.responses[message["id"]] = message
            else:
                # Handle server requests
                await self._handle_server_request(message)
        elif "method" in message:
            # Handle server notifications
            await self._handle_server_notification(message)

    async def _handle_server_request(self, message: dict):
        """Handle incoming requests from the server."""
        resp = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "error": {
                "code": -32601,
                "message": f"Method {message.get('method')} not implemented in client"
            }
        }
        await self._queue_message(resp)

    async def _handle_server_notification(self, message: dict):
        """Handle incoming notifications from the server."""
        method = message.get("method")
        if method:
            logger.debug(f"Server {self.server_name}: Received notification: {method}")

    async def _queue_message(self, message: dict) -> bool:
        """Queue a message for sending to avoid concurrent writes."""
        if self._shutdown:
            return False
        await self._message_queue.put(message)
        return True

    async def _wait_for_response(self, req_id: int) -> Optional[Dict]:
        """Wait for a response with timeout."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < self.timeout:
            if req_id in self.responses:
                resp = self.responses.pop(req_id)
                if "error" in resp:
                    logger.error(f"Server {self.server_name}: Error response: {resp['error']}")
                    return None
                return resp.get("result")
            await asyncio.sleep(0.01)
        logger.error(f"Server {self.server_name}: Request {req_id} timed out after {self.timeout}s")
        return None

    async def start(self) -> bool:
        """Start the MCP client and initialize the connection."""
        if self.process:
            return True

        # Expand path arguments
        expanded_args = [
            os.path.expanduser(a) if isinstance(a, str) and "~" in a else a
            for a in self.args
        ]

        # Prepare environment
        env_vars = os.environ.copy()
        if self.env:
            env_vars.update(self.env)

        try:
            # Start the process
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *expanded_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars
            )

            # Start message processing tasks
            self.receive_task = asyncio.create_task(self._receive_loop())
            self._queue_processor = asyncio.create_task(self._process_queue())

            # Initialize the connection
            return await self._perform_initialize()
        except Exception as e:
            logger.error(f"Server {self.server_name}: Failed to start: {str(e)}")
            await self.stop()
            return False

    async def _perform_initialize(self) -> bool:
        """Initialize the connection with the server."""
        self.request_id += 1
        req_id = self.request_id
        
        init_request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {"sampling": {}},
                "clientInfo": {
                    "name": "DolphinMCPClient",
                    "version": "1.0.0"
                }
            }
        }
        
        if not await self._queue_message(init_request):
            return False

        result = await self._wait_for_response(req_id)
        if not result:
            return False

        # Send initialized notification
        await self._queue_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })

        self.server_capabilities = result.get("capabilities", {})
        return True

    async def list_tools(self) -> List[Dict]:
        """Get list of available tools from the server."""
        if not self.process:
            return []

        self.request_id += 1
        req_id = self.request_id
        
        if not await self._queue_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {}
        }):
            return []

        result = await self._wait_for_response(req_id)
        if not result:
            return []

        self.tools = result.get("tools", [])
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Dict:
        """Call a tool on the server."""
        if not self.process:
            return {"error": "Not started"}

        self.request_id += 1
        req_id = self.request_id
        
        if not await self._queue_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }):
            return {"error": "Failed to queue message"}

        result = await self._wait_for_response(req_id)
        if not result:
            return {"error": f"Timeout waiting for tool result after {self.timeout}s"}
        
        return result

    async def stop(self):
        """Stop the client and cleanup resources."""
        async with self._cleanup_lock:
            if self._shutdown:
                return
            self._shutdown = True
            
            # Cancel background tasks
            if self.receive_task and not self.receive_task.done():
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass
            
            if hasattr(self, '_queue_processor') and not self._queue_processor.done():
                self._queue_processor.cancel()
                try:
                    await self._queue_processor
                except asyncio.CancelledError:
                    pass

            # Cleanup process
            if self.process:
                try:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Server {self.server_name}: Force killing process after timeout")
                        self.process.kill()
                        await self.process.wait()
                except Exception as e:
                    logger.error(f"Server {self.server_name}: Error during process cleanup: {str(e)}")
                finally:
                    if self.process.stdin:
                        self.process.stdin.close()
                    self.process = None

    async def close(self):
        """Alias for stop()."""
        await self.stop()
    
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()