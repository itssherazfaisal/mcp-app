"""
Example client to test the MCP server
"""
import httpx
import json
import asyncio

class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def call_mcp(self, method: str, params: dict = None, request_id: int = 1):
        """Make a call to the MCP server"""
        payload = {
            "method": method,
            "id": request_id
        }
        if params:
            payload["params"] = params
        
        response = await self.client.post(f"{self.base_url}/mcp", json=payload)
        return response.json()
    
    async def initialize(self):
        """Initialize the MCP connection"""
        return await self.call_mcp("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self):
        """List available tools"""
        return await self.call_mcp("tools/list")
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific tool"""
        return await self.call_mcp("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def list_resources(self):
        """List available resources"""
        return await self.call_mcp("resources/list")
    
    async def read_resource(self, uri: str):
        """Read a specific resource"""
        return await self.call_mcp("resources/read", {"uri": uri})
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

# Test function
async def test_mcp_server():
    client = MCPClient("http://localhost:8000")
    
    try:
        # Initialize
        print("=== Initializing ===")
        init_response = await client.initialize()
        print(json.dumps(init_response, indent=2))
        
        # List tools
        print("\n=== Listing Tools ===")
        tools_response = await client.list_tools()
        print(json.dumps(tools_response, indent=2))
        
        # Call calculator tool
        print("\n=== Calling Calculator Tool ===")
        calc_response = await client.call_tool("calculate", {
            "operation": "multiply",
            "a": 15,
            "b": 7
        })
        print(json.dumps(calc_response, indent=2))
        
        # Call time tool
        print("\n=== Calling Time Tool ===")
        time_response = await client.call_tool("current_time", {"format": "%Y-%m-%d %H:%M:%S"})
        print(json.dumps(time_response, indent=2))
        
        # Call text analyzer
        print("\n=== Calling Text Analyzer ===")
        text_response = await client.call_tool("analyze_text", {
            "text": "Hello world! This is a test message with multiple words and sentences."
        })
        print(json.dumps(text_response, indent=2))
        
        # List resources
        print("\n=== Listing Resources ===")
        resources_response = await client.list_resources()
        print(json.dumps(resources_response, indent=2))
        
        # Read resource
        print("\n=== Reading Server Info Resource ===")
        resource_response = await client.read_resource("server://info")
        print(json.dumps(resource_response, indent=2))
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
