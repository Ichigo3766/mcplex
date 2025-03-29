"""Type definitions and data structures for MCP."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import asyncio

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    init_timeout_multiplier: float = 1.0

@dataclass
class MCPConnection:
    """Represents an active MCP server connection."""
    config: ServerConfig
    process: Optional[asyncio.subprocess.Process] = None
    tools: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.tools = self.tools or []

@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    server_name: str
    description: str
    parameters: Dict[str, Any]

    @property
    def full_name(self) -> str:
        """Get the fully qualified tool name."""
        return f"{self.server_name}_{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary format."""
        return {
            "name": self.full_name,
            "description": self.description,
            "parameters": self.parameters
        }