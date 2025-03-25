"""
Connection pooling implementation for Dolphin MCP.
Provides efficient connection management and reuse for MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from .mcp_client import MCPClient

logger = logging.getLogger("dolphin_mcp")

class MCPConnectionPool:
    """
    A connection pool for MCP servers that manages connection lifecycle and reuse.
    
    Features:
    - Maintains a pool of connections per server
    - Limits maximum concurrent connections
    - Automatically creates new connections when needed
    - Handles connection cleanup and reuse
    """
    
    def __init__(self, max_connections: int = 10):
        """
        Initialize the connection pool.
        
        Args:
            max_connections: Maximum number of concurrent connections per server
        """
        self.max_connections = max_connections
        self.connections: Dict[str, List[MCPClient]] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.active_connections: Dict[str, int] = {}
        self._cleanup_lock = asyncio.Lock()
        self._shutdown = False
        self._health_check_tasks: Dict[str, asyncio.Task] = {}

    async def _check_connection_health(self, client: MCPClient) -> bool:
        """Check if a connection is healthy by attempting to list tools."""
        try:
            tools = await client.list_tools()
            return bool(tools)
        except Exception:
            return False

    async def _start_health_checks(self, server_name: str):
        """Start periodic health checks for a server's connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                if server_name not in self.connections:
                    break
                
                unhealthy = []
                for client in self.connections[server_name]:
                    if not await self._check_connection_health(client):
                        unhealthy.append(client)
                
                for client in unhealthy:
                    await client.stop()
                    self.connections[server_name].remove(client)
                    self.active_connections[server_name] -= 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {server_name}: {str(e)}")

    async def get_connection(self, server_name: str, config: dict) -> Optional[MCPClient]:
        """
        Get a connection from the pool or create a new one if needed.
        
        Args:
            server_name: Name of the MCP server
            config: Server configuration dictionary with optional 'timeout' setting
            
        Returns:
            An MCPClient instance or None if pool is shutdown
        """
        if self._shutdown:
            return None
            
        # Check if server is disabled
        if config.get("disabled", False):
            logger.info(f"Skipping disabled server: {server_name}")
            return None
            
        if server_name not in self.connections:
            self.connections[server_name] = []
            self.semaphores[server_name] = asyncio.Semaphore(self.max_connections)
            self.active_connections[server_name] = 0
            # Start health checks for this server
            self._health_check_tasks[server_name] = asyncio.create_task(
                self._start_health_checks(server_name)
            )
            
        async with self.semaphores[server_name]:
            # Try to reuse existing connection
            while self.connections[server_name]:
                client = self.connections[server_name].pop()
                if client.process and not client._shutdown:
                    # Verify connection health before reuse
                    if await self._check_connection_health(client):
                        self.active_connections[server_name] += 1
                        return client
                    else:
                        await client.stop()
                        self.active_connections[server_name] -= 1
                    
            # Create new connection if under limit
            if self.active_connections[server_name] < self.max_connections:
                try:
                    timeout = float(config.get("timeout", 3600.0))  # Default 1 hour timeout
                    client = MCPClient(
                        server_name=server_name,
                        command=config.get("command"),
                        args=config.get("args", []),
                        env=config.get("env", {}),
                        timeout=timeout
                    )
                    ok = await client.start()
                    if ok:
                        self.active_connections[server_name] += 1
                        return client
                except Exception as e:
                    logger.error(f"Error creating connection for {server_name}: {str(e)}")
                    
            return None

    async def release_connection(self, server_name: str, client: MCPClient):
        """
        Release a connection back to the pool.
        
        Args:
            server_name: Name of the MCP server
            client: The MCPClient instance to release
        """
        if self._shutdown:
            await client.stop()
            return
            
        # Check connection health before returning to pool
        if client.process and not client._shutdown and await self._check_connection_health(client):
            self.connections[server_name].append(client)
        else:
            await client.stop()
        self.active_connections[server_name] -= 1

    async def cleanup(self):
        """Clean up all connections and tasks in the pool."""
        async with self._cleanup_lock:
            if self._shutdown:
                return
                
            self._shutdown = True
            
            # Cancel health check tasks
            for task in self._health_check_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Clean up all connections
            cleanup_tasks = []
            for server_name, clients in self.connections.items():
                while clients:
                    client = clients.pop()
                    cleanup_tasks.append(client.stop())
                    
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
                
            # Clear all state
            self.connections.clear()
            self.semaphores.clear()
            self.active_connections.clear()
            self._health_check_tasks.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()