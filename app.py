#!/usr/bin/env python3
"""
A Python MCP (Model Context Protocol) Server implementation.
This server provides tools that can be called by OpenAI's models.
Fixed for proper JSON-RPC 2.0 compliance.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .tools import register_all_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: str
    version: str

class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.tool_handlers: Dict[str, callable] = {}
        self.resource_handlers: Dict[str, callable] = {}
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"Initialized MCP Server: {name} v{version}")
    
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
    
    async def handle_request(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle incoming MCP requests and return JSON-RPC 2.0 compliant response"""
        start_time = datetime.now()
        logger.info(f"Handling MCP request: {request.method} (ID: {request.id})")
        
        try:
            # Handle OpenAI method aliases
            method = request.method
            if method == "openai.tool_list":
                method = "tools/list"
            elif method == "openai.tool_call":
                method = "tools/call"
            if method == "initialize":
                result = await self._handle_initialize(request)
            elif method == "tools/list":
                result = await self._handle_tools_list(request)
            elif method == "tools/call":
                result = await self._handle_tools_call(request)
            elif method == "resources/list":
                result = await self._handle_resources_list(request)
            elif method == "resources/read":
                result = await self._handle_resources_read(request)
            else:
                logger.warning(f"Unknown method: {request.method}")
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {request.method}"
                    },
                    "id": request.id
                }
            
            response = {
                "jsonrpc": "2.0",
                "result": result,
                "id": request.id
            }
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Request {request.method} completed successfully in {duration:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error handling request {request.method}: {str(e)}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": request.id
            }
    
    async def _handle_initialize(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def _handle_tools_list(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle tools list request"""
        tools_list = []
        for tool in self.tools.values():
            tool_dict = asdict(tool)
            tools_list.append(tool_dict)
        
        logger.info(f"Returning {len(tools_list)} tools")
        return {"tools": tools_list}
    
    async def _handle_tools_call(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle tool call request"""
        if not request.params or "name" not in request.params:
            raise ValueError("Missing tool name in request")
        
        tool_name = request.params["name"]
        arguments = request.params.get("arguments", {})
        
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            handler = self.tool_handlers[tool_name]
            result = await handler(arguments) if asyncio.iscoroutinefunction(handler) else handler(arguments)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ]
            }
        except Exception as e:
            raise ValueError(f"Tool execution failed: {str(e)}")
    
    async def _handle_resources_list(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle resources list request"""
        resources_list = [asdict(resource) for resource in self.resources.values()]
        return {"resources": resources_list}
    
    async def _handle_resources_read(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle resource read request"""
        if not request.params or "uri" not in request.params:
            raise ValueError("Missing resource URI in request")
        
        uri = request.params["uri"]
        
        if uri not in self.resource_handlers:
            raise ValueError(f"Resource not found: {uri}")
        
        try:
            handler = self.resource_handlers[uri]
            content = await handler() if asyncio.iscoroutinefunction(handler) else handler()
            
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": self.resources[uri].mimeType,
                        "text": content
                    }
                ]
            }
        except Exception as e:
            raise ValueError(f"Resource read failed: {str(e)}")

# Create the MCP server instance
mcp_server = MCPServer("ByteGenie", "1.0.0")


# Example resource implementations
def get_server_info() -> str:
    """Get server information"""
    try:
        uptime = datetime.now(timezone.utc) - mcp_server.start_time
        info = {
            "name": mcp_server.name,
            "version": mcp_server.version,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools_count": len(mcp_server.tools),
            "resources_count": len(mcp_server.resources),
            "available_tools": list(mcp_server.tools.keys())
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Server info error: {str(e)}"

# Register resources
mcp_server.add_resource(
    uri="server://info",
    name="Server Information",
    description="Current server status and information",
    mime_type="application/json",
    handler=get_server_info
)
register_all_tools(mcp_server)

# FastAPI app
app = FastAPI(
    title="ByteGenie MCP Server", 
    description="Model Context Protocol Server for ByteGenie - JSON-RPC 2.0 Compliant",
    version=mcp_server.version
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint - JSON-RPC 2.0 compliant"""
    start_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Parse request body
        body = await request.json()
        logger.info(f"Raw MCP request from {client_ip}: {json.dumps(body, indent=2)}")
        
        # Validate JSON-RPC 2.0 format
        if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
            logger.error("Invalid JSON-RPC version")
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request - missing or invalid jsonrpc version"
                    },
                    "id": body.get("id")
                },
                status_code=400
            )
        
        if "method" not in body:
            logger.error("Missing method in request")
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request - missing method"
                    },
                    "id": body.get("id")
                },
                status_code=400
            )
        
        # Create MCP request object
        mcp_request = MCPRequest(**body)
        
        # Handle the request
        response = await mcp_server.handle_request(mcp_request)
        
        # Log successful responses
        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"MCP response sent in {duration:.2f}ms: {json.dumps(response, indent=2)}")
        
        return JSONResponse(content=response)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from {client_ip}: {e}")
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error - Invalid JSON"
                },
                "id": None
            },
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error processing MCP request from {client_ip}: {e}", exc_info=True)
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal server error: {str(e)}"
                },
                "id": body.get("id") if 'body' in locals() else None
            },
            status_code=500
        )

@app.get("/")
async def root():
    """Root endpoint with server info"""
    uptime = datetime.now(timezone.utc) - mcp_server.start_time
    return {
        "name": mcp_server.name,
        "version": mcp_server.version,
        "protocol": "MCP (Model Context Protocol)",
        "jsonrpc": "2.0",
        "status": "running",
        "uptime_seconds": uptime.total_seconds(),
        "endpoints": {
            "mcp": "/mcp",
            "health": "/health",
            "docs": "/docs",
            "ping": "/ping"
        },
        "tools_available": len(mcp_server.tools),
        "resources_available": len(mcp_server.resources),
        "available_tools": list(mcp_server.tools.keys()),
        "available_resources": list(mcp_server.resources.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now(timezone.utc) - mcp_server.start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime=str(uptime),
        version=mcp_server.version
    )

@app.get("/ping")
async def ping():
    """Simple ping endpoint for monitoring"""
    return {
        "status": "pong",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": mcp_server.name
    }

# Test endpoint to verify tools/list functionality
@app.get("/test-tools")
async def test_tools():
    """Test endpoint to verify tools list"""
    try:
        fake_request = MCPRequest(method="tools/list", id="test")
        response = await mcp_server.handle_request(fake_request)
        return response
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting {mcp_server.name} v{mcp_server.version}")
    logger.info(f"Available tools: {list(mcp_server.tools.keys())}")
    logger.info(f"Available resources: {list(mcp_server.resources.keys())}")
    logger.info(f"Server will listen on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )
