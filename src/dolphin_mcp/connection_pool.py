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

    async def get_connection(self, server_name: str, config: dict) -> Optional[MCPClient]:
        """
        Get a connection from the pool or create a new one if needed.
        
        Args:
            server_name: Name of the MCP server
            config: Server configuration dictionary
            
        Returns:
            An MCPClient instance or None if pool is shutdown
        """
        if self._shutdown:
            return None
            
        if server_name not in self.connections:
            self.connections[server_name] = []
            self.semaphores[server_name] = asyncio.Semaphore(self.max_connections)
            self.active_connections[server_name] = 0
            
        async with self.semaphores[server_name]:
            # Try to reuse existing connection
            while self.connections[server_name]:
                client = self.connections[server_name].pop()
                if client.process and not client._shutdown:
                    self.active_connections[server_name] += 1
                    return client
                    
            # Create new connection if under limit
            if self.active_connections[server_name] < self.max_connections:
                try:
                    client = MCPClient(
                        server_name=server_name,
                        command=config.get("command"),
                        args=config.get("args", []),
                        env=config.get("env", {})
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
            
        if client.process and not client._shutdown:
            self.connections[server_name].append(client)
        else:
            await client.stop()
        self.active_connections[server_name] -= 1

    async def cleanup(self):
        """Clean up all connections in the pool."""
        async with self._cleanup_lock:
            if self._shutdown:
                return
                
            self._shutdown = True
            cleanup_tasks = []
            
            for server_name, clients in self.connections.items():
                while clients:
                    client = clients.pop()
                    cleanup_tasks.append(client.stop())
                    
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
                
            self.connections.clear()
            self.semaphores.clear()
            self.active_connections.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()