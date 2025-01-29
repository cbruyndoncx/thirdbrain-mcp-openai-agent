class BaseMCPError(Exception):
    """Base class for all MCP-related errors."""
    pass

class ConfigurationError(BaseMCPError):
    """Raised when there is a configuration-related error."""
    pass

class ConnectionError(BaseMCPError):
    """Raised when there is a connection-related error."""
    pass

class ToolError(BaseMCPError):
    """Raised when there is an error related to tool operations."""
    pass

class DatabaseConnectionError(BaseMCPError):
    """Raised when there is a database connection-related error."""
    pass
