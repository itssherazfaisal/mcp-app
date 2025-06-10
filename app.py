#!/usr/bin/env python3
"""
A Python MCP (Model Context Protocol) Server implementation.
This server provides tools that can be called by OpenAI's models.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Protocol Data Structures
@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class Resource:
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class MCPResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.tool_handlers: Dict[str, callable] = {}
        self.resource_handlers: Dict[str, callable] = {}
        
    def add_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: callable):
        """Add a tool to the MCP server"""
        tool = Tool(name=name, description=description, inputSchema=input_schema)
        self.tools[name] = tool
        self.tool_handlers[name] = handler
        logger.info(f"Added tool: {name}")
        
    def add_resource(self, uri: str, name: str, description: str, mime_type: str, handler: callable):
        """Add a resource to the MCP server"""
        resource = Resource(uri=uri, name=name, description=description, mimeType=mime_type)
        self.resources[uri] = resource
        self.resource_handlers[uri] = handler
        logger.info(f"Added resource: {uri}")
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP requests"""
        try:
            if request.method == "initialize":
                return await self._handle_initialize(request)
            elif request.method == "tools/list":
                return await self._handle_tools_list(request)
            elif request.method == "tools/call":
                return await self._handle_tools_call(request)
            elif request.method == "resources/list":
                return await self._handle_resources_list(request)
            elif request.method == "resources/read":
                return await self._handle_resources_read(request)
            else:
                return MCPResponse(
                    error={"code": -32601, "message": f"Method not found: {request.method}"},
                    id=request.id
                )
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
                id=request.id
            )
    
    async def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """Handle initialization request"""
        return MCPResponse(
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True}
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            },
            id=request.id
        )
    
    async def _handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        """Handle tools list request"""
        tools_list = [asdict(tool) for tool in self.tools.values()]
        return MCPResponse(
            result={"tools": tools_list},
            id=request.id
        )
    
    async def _handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        """Handle tool call request"""
        if not request.params or "name" not in request.params:
            return MCPResponse(
                error={"code": -32602, "message": "Missing tool name"},
                id=request.id
            )
        
        tool_name = request.params["name"]
        arguments = request.params.get("arguments", {})
        
        if tool_name not in self.tool_handlers:
            return MCPResponse(
                error={"code": -32601, "message": f"Tool not found: {tool_name}"},
                id=request.id
            )
        
        try:
            handler = self.tool_handlers[tool_name]
            result = await handler(arguments) if asyncio.iscoroutinefunction(handler) else handler(arguments)
            
            return MCPResponse(
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                },
                id=request.id
            )
        except Exception as e:
            return MCPResponse(
                error={"code": -32603, "message": f"Tool execution failed: {str(e)}"},
                id=request.id
            )
    
    async def _handle_resources_list(self, request: MCPRequest) -> MCPResponse:
        """Handle resources list request"""
        resources_list = [asdict(resource) for resource in self.resources.values()]
        return MCPResponse(
            result={"resources": resources_list},
            id=request.id
        )
    
    async def _handle_resources_read(self, request: MCPRequest) -> MCPResponse:
        """Handle resource read request"""
        if not request.params or "uri" not in request.params:
            return MCPResponse(
                error={"code": -32602, "message": "Missing resource URI"},
                id=request.id
            )
        
        uri = request.params["uri"]
        
        if uri not in self.resource_handlers:
            return MCPResponse(
                error={"code": -32601, "message": f"Resource not found: {uri}"},
                id=request.id
            )
        
        try:
            handler = self.resource_handlers[uri]
            content = await handler() if asyncio.iscoroutinefunction(handler) else handler()
            
            return MCPResponse(
                result={
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": self.resources[uri].mimeType,
                            "text": content
                        }
                    ]
                },
                id=request.id
            )
        except Exception as e:
            return MCPResponse(
                error={"code": -32603, "message": f"Resource read failed: {str(e)}"},
                id=request.id
            )

# Create the MCP server instance
mcp_server = MCPServer("example-mcp-server", "1.0.0")

# Example tool implementations
def calculate_tool(args: Dict[str, Any]) -> str:
    """Example calculator tool"""
    operation = args.get("operation", "add")
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return f"Result: {result}"

def current_time_tool(args: Dict[str, Any]) -> str:
    """Get current time"""
    format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime(format_str)

def text_analyzer_tool(args: Dict[str, Any]) -> str:
    """Analyze text properties"""
    text = args.get("text", "")
    analysis = {
        "length": len(text),
        "words": len(text.split()),
        "lines": len(text.split('\n')),
        "characters_no_spaces": len(text.replace(' ', ''))
    }
    return json.dumps(analysis, indent=2)

# Example resource implementations
def get_server_info() -> str:
    """Get server information"""
    info = {
        "name": mcp_server.name,
        "version": mcp_server.version,
        "uptime": "Running",
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(info, indent=2)

# Register tools
mcp_server.add_tool(
    name="calculate",
    description="Perform basic mathematical calculations",
    input_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The mathematical operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number", 
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    },
    handler=calculate_tool
)

mcp_server.add_tool(
    name="current_time",
    description="Get the current date and time",
    input_schema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Time format string (default: %Y-%m-%d %H:%M:%S)"
            }
        }
    },
    handler=current_time_tool
)

mcp_server.add_tool(
    name="analyze_text",
    description="Analyze text and return statistics",
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to analyze"
            }
        },
        "required": ["text"]
    },
    handler=text_analyzer_tool
)

# Register resources
mcp_server.add_resource(
    uri="server://info",
    name="Server Information",
    description="Current server status and information",
    mime_type="application/json",
    handler=get_server_info
)

# FastAPI app
app = FastAPI(title="MCP Server", description="Model Context Protocol Server")

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint"""
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
        response = await mcp_server.handle_request(mcp_request)
        return JSONResponse(content=response.dict(exclude_none=True))
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}")
        return JSONResponse(
            content={"error": {"code": -32700, "message": "Parse error"}},
            status_code=400
        )

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "name": mcp_server.name,
        "version": mcp_server.version,
        "protocol": "MCP",
        "endpoints": {
            "mcp": "/mcp"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
