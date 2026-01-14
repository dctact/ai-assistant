"""
Tool manager for integrating external MCP tools with the AI assistant

This module manages the lifecycle of HTTP connections to the MCP server
and provides tool detection and execution capabilities.

Overview:
- Async context manager for proper resource lifecycle
- HTTP client management with automatic cleanup
- Tool pattern detection and execution
- Graceful error handling with resource cleanup

Key Dependencies:
- httpx for async HTTP client (MUST be properly closed)
- MCP server for tool discovery and execution

Resource Management:
- Use 'async with ToolManager()' for automatic cleanup
- Client is closed on error during initialization
- Fallback __del__ warns if cleanup was missed

Recent Changes:
- 2025-01-14: Added async context manager protocol for proper resource cleanup
- 2025-01-14: Added cleanup on initialization failure to prevent leaks
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import httpx
from loguru import logger


class ToolManager:
    """Manages external tool integration for the AI assistant.

    This class implements the async context manager protocol to ensure
    proper cleanup of HTTP client resources even in error scenarios.

    Usage:
        # Preferred: Use async context manager for automatic cleanup
        async with ToolManager() as tool_manager:
            result = await tool_manager.execute_tool("web_search", query="test")

        # Alternative: Manual lifecycle management
        tool_manager = ToolManager()
        try:
            await tool_manager.initialize()
            result = await tool_manager.execute_tool("web_search", query="test")
        finally:
            await tool_manager.close()
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        """Initialize tool manager.

        Note: HTTP client is created here but must be explicitly closed
        via close() method or by using the async context manager protocol.
        """
        self.mcp_server_url = mcp_server_url.rstrip("/")
        # Create client with timeout to prevent hanging connections
        self.client: Optional[httpx.AsyncClient] = httpx.AsyncClient(timeout=30.0)
        self.available_tools = []
        self._initialized = False
        self._closed = False

        # Tool detection patterns
        self.tool_patterns = {
            "web_search": [
                r"search (?:the )?web for",
                r"look up",
                r"find (?:information )?about",
                r"what(?:'s| is) (?:the )?latest",
                r"current (?:news|events)",
                r"google|search|find online"
            ],
            "weather": [
                r"weather in",
                r"temperature in",
                r"how(?:'s| is) the weather",
                r"forecast for",
                r"climate in",
                r"is it (?:raining|sunny|cloudy)"
            ],
            "calculate": [
                r"calculate|compute|math|equation",
                r"\d+\s*[+\-*/]\s*\d+",
                r"what(?:'s| is) \d+",
                r"square root|factorial|percentage"
            ],
            "news_search": [
                r"latest news",
                r"news about",
                r"recent articles",
                r"what(?:'s| is) happening",
                r"current events"
            ]
        }
    
    async def __aenter__(self):
        """Async context manager entry.

        Automatically initializes the tool manager when entering
        the async context. If initialization fails, cleanup is
        automatic via __aexit__.
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Ensures HTTP client is always closed, even if an exception
        occurred during tool operations.
        """
        await self.close()
        # Don't suppress exceptions
        return False

    async def initialize(self):
        """Initialize the tool manager by loading available tools.

        If initialization fails, the HTTP client is automatically closed
        to prevent resource leaks. This ensures that even if the MCP server
        is unreachable, we don't leave hanging connections.
        """
        if self._initialized:
            logger.debug("Tool manager already initialized")
            return

        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            await self._load_available_tools()
            self._initialized = True
            logger.info(f"Tool manager initialized with {len(self.available_tools)} tools")
        except Exception as e:
            # Critical: Clean up HTTP client on initialization failure
            # This prevents resource leaks when MCP server is unreachable
            logger.error(f"Failed to initialize tool manager: {e}")
            await self.close()
            raise
    
    async def _load_available_tools(self):
        """Load available tools from MCP server"""
        try:
            response = await self.client.get(f"{self.mcp_server_url}/tools/list")
            if response.status_code == 200:
                data = response.json()
                self.available_tools = data.get("tools", [])
            else:
                logger.warning(f"Failed to load tools: {response.status_code}")
        except Exception as e:
            logger.error(f"Error loading tools: {e}")
    
    def detect_tool_needs(self, message: str) -> List[Dict[str, Any]]:
        """Detect if a message requires tool usage"""
        message_lower = message.lower()
        detected_tools = []
        
        for tool_name, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected_tools.append({
                        "tool": tool_name,
                        "pattern": pattern,
                        "confidence": 0.8  # Could be enhanced with ML
                    })
                    break  # One match per tool is enough
        
        return detected_tools
    
    async def execute_tool(self, tool_name: str, **params) -> Dict[str, Any]:
        """Execute a tool via MCP server"""
        try:
            payload = {"tool_name": tool_name, **params}
            
            response = await self.client.post(
                f"{self.mcp_server_url}/tools/execute",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Tool execution failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def extract_search_query(self, message: str) -> str:
        """Extract search query from user message"""
        # Simple extraction - could be enhanced
        patterns = [
            r"search (?:the )?web for (.+)",
            r"look up (.+)",
            r"find (?:information )?about (.+)",
            r"what(?:'s| is) (?:the )?latest (?:on )?(.+)",
            r"news about (.+)"
        ]
        
        message_lower = message.lower()
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: return the whole message cleaned up
        return re.sub(r'^(?:search|find|look up|google)\s+', '', message_lower).strip()
    
    def extract_location(self, message: str) -> str:
        """Extract location from weather-related message"""
        patterns = [
            r"weather in (.+)",
            r"temperature in (.+)",
            r"forecast for (.+)",
            r"climate in (.+)"
        ]
        
        message_lower = message.lower()
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback patterns
        if "weather" in message_lower:
            words = message.split()
            # Look for location indicators
            location_words = []
            found_in = False
            for word in words:
                if word.lower() in ["in", "for", "at"]:
                    found_in = True
                elif found_in and word.lower() not in ["the", "weather", "forecast"]:
                    location_words.append(word)
            
            if location_words:
                return " ".join(location_words)
        
        return "current location"  # Default fallback
    
    def extract_calculation(self, message: str) -> str:
        """Extract mathematical expression from message"""
        # Look for mathematical expressions
        math_patterns = [
            r"calculate (.+)",
            r"compute (.+)",
            r"what(?:'s| is) (.+)",
            r"(\d+(?:\.\d+)?\s*[+\-*/^]\s*\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*\d+(?:\.\d+)?)*)"
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, message)
            if match:
                expr = match.group(1).strip()
                # Clean up common words
                expr = re.sub(r'\b(?:plus|minus|times|divided by|to the power of)\b', 
                            lambda m: {
                                'plus': '+', 'minus': '-', 'times': '*', 
                                'divided by': '/', 'to the power of': '**'
                            }[m.group()], expr)
                return expr
        
        return message.strip()
    
    async def process_message_with_tools(self, message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a message and execute any needed tools"""
        detected_tools = self.detect_tool_needs(message)
        tool_results = []
        
        for tool_info in detected_tools:
            tool_name = tool_info["tool"]
            
            try:
                if tool_name == "web_search":
                    query = self.extract_search_query(message)
                    result = await self.execute_tool("web_search", query=query, count=5)
                    tool_results.append({"tool": "web_search", "query": query, "result": result})
                
                elif tool_name == "weather":
                    location = self.extract_location(message)
                    result = await self.execute_tool("weather", location=location)
                    tool_results.append({"tool": "weather", "location": location, "result": result})
                
                elif tool_name == "calculate":
                    expression = self.extract_calculation(message)
                    result = await self.execute_tool("calculate", expression=expression)
                    tool_results.append({"tool": "calculate", "expression": expression, "result": result})
                
                elif tool_name == "news_search":
                    query = self.extract_search_query(message)
                    result = await self.execute_tool("news_search", query=query, count=5)
                    tool_results.append({"tool": "news_search", "query": query, "result": result})
                    
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                tool_results.append({
                    "tool": tool_name, 
                    "error": str(e)
                })
        
        # Enhance the message with tool results
        enhanced_message = message
        if tool_results:
            enhanced_message += "\n\n[Tool Results]\n"
            for tool_result in tool_results:
                if "error" in tool_result:
                    enhanced_message += f"- {tool_result['tool']}: Error - {tool_result['error']}\n"
                else:
                    enhanced_message += f"- {tool_result['tool']}: {json.dumps(tool_result['result'], indent=2)}\n"
        
        return enhanced_message, tool_results
    
    async def close(self):
        """Close the HTTP client and free resources.

        This method is idempotent - calling it multiple times is safe.
        It checks if the client exists and hasn't already been closed
        before attempting to close it.
        """
        if self._closed:
            return

        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.debug("HTTP client closed successfully")

        self.client = None
        self._closed = True

    def __del__(self):
        """Destructor to warn about unclosed resources.

        This is a fallback safety net. Ideally, resources should be
        cleaned up explicitly via close() or async context manager.

        We only warn here because __del__ is called during garbage
        collection, and we can't await async cleanup at that point.
        """
        if self.client and not self._closed:
            logger.warning(
                "ToolManager not properly closed. Resources may leak. "
                "Use 'async with ToolManager()' or call close() explicitly."
            )


# Global tool manager instance
_tool_manager = None

async def get_tool_manager() -> ToolManager:
    """Get or create the global tool manager instance"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
        await _tool_manager.initialize()
    return _tool_manager