"""
FastAPI-based MCP server for AI Assistant

Provides REST API and WebSocket endpoints for agent interactions.

Overview:
- RESTful API for agent queries, memory management, and tool execution
- WebSocket support for real-time streaming interactions
- JWT and API key authentication with rate limiting
- Integration with Ollama for LLM operations

Key Dependencies:
- FastAPI for API framework
- slowapi for rate limiting (prevents brute force attacks)
- DatabaseManager for persistent storage
- OllamaClient for LLM interactions

Security Features:
- Rate limiting on authentication endpoints (5 attempts per 15 minutes)
- JWT token validation
- API key authentication
- CORS middleware for web clients

Recent Changes:
- 2025-10-27: Added rate limiting to prevent brute force attacks on auth endpoints
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import DatabaseManager, VectorStore
from .models import *
from .auth import AuthMiddleware, get_current_user, get_read_user, get_write_user, get_admin_user, login
from .ollama_client import OllamaClient
from .tools import tool_registry


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> [connection_ids]
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Connect a new WebSocket"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user: {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Disconnect a WebSocket"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id] = [
                cid for cid in self.user_connections[user_id] if cid != connection_id
            ]
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} for user: {user_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
    
    async def send_user_message(self, message: str, user_id: str):
        """Send message to all connections for a user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection_id in self.active_connections:
            await self.send_personal_message(message, connection_id)


# Global instances
db_manager = None
vector_store = None
ollama_client = None
connection_manager = ConnectionManager()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager, vector_store, ollama_client
    
    # Startup
    logger.info("Starting MCP server...")
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize vector store
    vector_store = VectorStore(db_manager)
    
    # Initialize Ollama client
    ollama_client = OllamaClient()
    await ollama_client.initialize()
    
    logger.info("MCP server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP server...")
    
    if ollama_client:
        await ollama_client.close()
    
    if db_manager:
        db_manager.close()
    
    logger.info("MCP server shut down")


def create_app() -> FastAPI:
    """Create FastAPI application with security middleware.

    Configures:
    - Rate limiting to prevent brute force attacks
    - CORS for cross-origin requests
    - Authentication middleware for all protected endpoints
    """
    app = FastAPI(
        title="AI Assistant MCP Server",
        description="FastAPI-based MCP server for AI Assistant with REST API and WebSocket support",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure rate limiter
    # Uses client IP address to track request rates
    # This prevents brute force attacks on authentication endpoints
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled for authentication endpoints")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add authentication middleware
    app.add_middleware(AuthMiddleware)

    return app


app = create_app()


# Health check and status endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_status = "healthy" if db_manager and await db_manager.health_check() else "unhealthy"
        
        # Check Ollama
        ollama_status = "healthy" if ollama_client and await ollama_client.health_check() else "unhealthy"
        
        # Calculate uptime
        uptime = time.time() - start_time
        
        # Get basic memory info
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent()
        }
        
        overall_status = "healthy" if db_status == "healthy" and ollama_status == "healthy" else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            database_status=db_status,
            ollama_status=ollama_status,
            memory_usage=memory_usage,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            database_status="unknown",
            ollama_status="unknown"
        )


@app.get("/status")
async def get_status(current_user: Dict[str, Any] = Depends(get_read_user)):
    """Get detailed server status"""
    try:
        # Get database stats
        db_stats = await db_manager.get_stats() if db_manager else {}
        
        # Get Ollama models
        models = ollama_client.available_models if ollama_client else []
        
        # Get connection info
        connection_info = {
            "active_websockets": len(connection_manager.active_connections),
            "connected_users": len(connection_manager.user_connections)
        }
        
        return {
            "status": "online",
            "uptime": time.time() - start_time,
            "database": db_stats,
            "models": models,
            "connections": connection_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


# Authentication endpoints
@app.post("/auth/login", response_model=AuthResponse)
@app.state.limiter.limit("5/15minutes")
async def login_endpoint(request: Request, auth_request: AuthRequest):
    """User login endpoint with rate limiting.

    Rate limit: 5 attempts per 15 minutes per IP address.

    This prevents brute force attacks where attackers try many
    passwords rapidly. After 5 failed attempts, the IP is blocked
    for 15 minutes. Legitimate users can retry after the window.

    The limit applies per IP, so shared IPs (corporate networks)
    may affect multiple users. Consider adding CAPTCHA after
    3 failures for better UX.
    """
    # Log suspicious activity - many failed attempts
    logger.info(f"Login attempt from {request.client.host}")
    return await login(auth_request)


@app.post("/auth/refresh", response_model=AuthResponse)
@app.state.limiter.limit("10/15minutes")
async def refresh_token(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Refresh JWT token with rate limiting.

    Rate limit: 10 attempts per 15 minutes per IP address.

    Token refresh is slightly more permissive than login (10 vs 5)
    because legitimate users may need to refresh more frequently,
    especially in long sessions or with short token expiry times.
    """
    # This would typically validate a refresh token
    # For now, just return the same token info
    logger.info(f"Token refresh for user: {current_user.get('user_id')}")
    return AuthResponse(
        access_token="refreshed_token",
        expires_in=1800,
        user_id=current_user.get("user_id")
    )


# Agent interaction endpoints
@app.post("/agent/query", response_model=QueryResponse)
async def agent_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Query the AI agent"""
    try:
        # Get conversation context if needed
        context_messages = []
        if request.conversation_id and request.context_length > 0:
            context_messages = await db_manager.get_conversation_context(
                request.conversation_id,
                limit=request.context_length
            )
        
        # Generate response using Ollama
        response = await ollama_client.generate(request, context_messages)
        
        # Save conversation to database
        if request.conversation_id:
            # Save user message
            await db_manager.add_message(
                request.conversation_id,
                MessageRole.USER,
                request.message,
                metadata=request.metadata
            )
            
            # Save assistant response
            await db_manager.add_message(
                request.conversation_id,
                MessageRole.ASSISTANT,
                response.response,
                metadata={"model": request.model, "tokens": response.tokens_used}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/agent/stream")
async def agent_stream(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Stream AI agent response"""
    try:
        # Get conversation context
        context_messages = []
        if request.conversation_id and request.context_length > 0:
            context_messages = await db_manager.get_conversation_context(
                request.conversation_id,
                limit=request.context_length
            )
        
        async def generate_stream():
            full_response = ""
            
            async for chunk in ollama_client.generate_stream(request, context_messages):
                full_response += chunk.chunk
                
                # Send chunk as SSE
                yield f"data: {chunk.json()}\n\n"
                
                if chunk.is_final:
                    break
            
            # Save conversation to database
            if request.conversation_id:
                # Save user message
                await db_manager.add_message(
                    request.conversation_id,
                    MessageRole.USER,
                    request.message,
                    metadata=request.metadata
                )
                
                # Save assistant response
                await db_manager.add_message(
                    request.conversation_id,
                    MessageRole.ASSISTANT,
                    full_response,
                    metadata={"model": request.model, "streamed": True}
                )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Agent stream failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream failed: {str(e)}")


# Memory endpoints
@app.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(
    request: MemorySearchRequest,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Search memory/conversation history"""
    try:
        start_time = time.time()
        
        if request.search_type == "fts":
            # Full-text search
            results = await db_manager.search_messages(
                request.query,
                limit=request.limit,
                conversation_id=request.conversation_id,
                start_date=request.start_date,
                end_date=request.end_date
            )
        elif request.search_type == "semantic":
            # Semantic search using vector store
            results = await vector_store.search_similar(
                request.query,
                limit=request.limit,
                threshold=request.similarity_threshold
            )
        elif request.search_type == "hybrid":
            # Hybrid search combining FTS and semantic
            fts_results = await db_manager.search_messages(
                request.query,
                limit=request.limit // 2,
                conversation_id=request.conversation_id
            )
            semantic_results = await vector_store.search_similar(
                request.query,
                limit=request.limit // 2,
                threshold=request.similarity_threshold
            )
            # Combine and deduplicate results
            results = fts_results + semantic_results
            # Remove duplicates based on message_id
            seen = set()
            unique_results = []
            for result in results:
                if result["message_id"] not in seen:
                    seen.add(result["message_id"])
                    unique_results.append(result)
            results = unique_results[:request.limit]
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        search_time = time.time() - start_time
        
        # Convert results to response format
        search_results = []
        for result in results:
            search_results.append(MemorySearchResult(
                message_id=result["message_id"],
                conversation_id=result["conversation_id"],
                role=MessageRole(result["role"]),
                content=result["content"],
                timestamp=result["timestamp"],
                similarity_score=result.get("similarity_score"),
                metadata=result.get("metadata", {})
            ))
        
        return MemorySearchResponse(
            results=search_results,
            total_count=len(search_results),
            search_time=search_time,
            query=request.query,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/memory/save", response_model=MemorySaveResponse)
async def memory_save(
    request: MemorySaveRequest,
    current_user: Dict[str, Any] = Depends(get_write_user)
):
    """Save message to memory"""
    try:
        # Save message to database
        message_id = await db_manager.add_message(
            request.conversation_id,
            request.role,
            request.content,
            message_type=request.message_type.value,
            metadata=request.metadata
        )
        
        # Generate and store embedding if not provided
        if not request.embedding and vector_store:
            try:
                embedding = await ollama_client.embed(request.content)
                await vector_store.store_embedding(
                    "message",
                    message_id,
                    embedding,
                    {"content": request.content, "role": request.role.value}
                )
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        elif request.embedding and vector_store:
            await vector_store.store_embedding(
                "message",
                message_id,
                request.embedding,
                {"content": request.content, "role": request.role.value}
            )
        
        return MemorySaveResponse(
            message_id=message_id,
            conversation_id=request.conversation_id,
            timestamp=datetime.utcnow(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Memory save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


# Project endpoints
@app.get("/projects/list", response_model=ProjectListResponse)
async def projects_list(
    page: int = 1,
    page_size: int = 20,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """List projects"""
    try:
        projects = await db_manager.get_projects(
            offset=(page - 1) * page_size,
            limit=page_size
        )
        
        total_count = await db_manager.get_projects_count()
        
        project_list = []
        for project in projects:
            project_list.append(ProjectInfo(
                project_id=project["project_id"],
                name=project["name"],
                description=project.get("description"),
                status=project.get("status", "active"),
                created_at=project["created_at"],
                updated_at=project["updated_at"],
                tags=project.get("tags", []),
                metadata=project.get("metadata", {})
            ))
        
        return ProjectListResponse(
            projects=project_list,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Projects list failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@app.post("/projects/create")
async def create_project(
    project_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_write_user)
):
    """Create a new project"""
    try:
        project_id = await db_manager.create_project(
            name=project_data["name"],
            description=project_data.get("description"),
            metadata=project_data.get("metadata", {})
        )
        
        return {"project_id": project_id, "success": True}
        
    except Exception as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


# WebSocket endpoint
@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str, token: str = Query(...)):
    """WebSocket endpoint for real-time interactions with authentication.

    Security: Requires a valid JWT token to establish WebSocket connection.
    This prevents unauthorized access to real-time streaming features.

    Args:
        websocket: WebSocket connection object
        connection_id: Unique connection identifier
        token: JWT authentication token (required query parameter)

    Token validation:
    - Verified before accepting WebSocket connection
    - Checked periodically during long-running connections
    - Connection closed if token expires or becomes invalid

    Connection flow:
    1. Validate JWT token
    2. Check user permissions
    3. Accept WebSocket connection
    4. Associate connection with user
    5. Process messages with user context
    6. Periodically revalidate token
    7. Clean disconnect on token expiry
    """
    # Validate authentication token before accepting WebSocket connection
    # This prevents unauthorized users from establishing connections
    try:
        from .auth import auth_manager
        payload = auth_manager.verify_token(token)
        user_id = payload.get("sub")
        permissions = payload.get("permissions", [])

        # Check if user has required permissions for WebSocket access
        if "read" not in permissions:
            logger.warning(f"WebSocket connection denied for {user_id}: missing read permission")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient permissions")
            return

        logger.info(f"WebSocket authentication successful for user: {user_id}")

    except HTTPException as e:
        # Authentication failed - reject connection
        logger.warning(f"WebSocket authentication failed for connection {connection_id}: {e.detail}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    # Accept connection only after successful authentication
    await connection_manager.connect(websocket, connection_id, user_id)

    # Track messages for periodic token revalidation
    # This prevents token expiry during long-running WebSocket sessions
    message_count = 0
    TOKEN_REVALIDATION_INTERVAL = 10  # Revalidate every 10 messages

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Periodically revalidate token during long connections
            # This ensures expired tokens don't remain connected
            message_count += 1
            if message_count % TOKEN_REVALIDATION_INTERVAL == 0:
                try:
                    auth_manager.verify_token(token)
                    logger.debug(f"Token revalidation successful for user {user_id} (message {message_count})")
                except HTTPException:
                    # Token expired or invalid - close connection gracefully
                    logger.warning(f"Token expired for user {user_id}, closing WebSocket")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Token expired. Please reconnect with a valid token.",
                        "code": "TOKEN_EXPIRED"
                    }))
                    break

            try:
                message = json.loads(data)
                message_obj = WebSocketMessage(**message)
                
                # Handle different message types
                if message_obj.type == "query":
                    # Handle query through WebSocket
                    request = QueryRequest(
                        message=message_obj.content,
                        conversation_id=message_obj.conversation_id,
                        stream=True
                    )
                    
                    # Get context
                    context_messages = []
                    if request.conversation_id:
                        context_messages = await db_manager.get_conversation_context(
                            request.conversation_id,
                            limit=10
                        )
                    
                    # Stream response
                    async for chunk in ollama_client.generate_stream(request, context_messages):
                        await websocket.send_text(json.dumps({
                            "type": "response_chunk",
                            "content": chunk.chunk,
                            "is_final": chunk.is_final,
                            "conversation_id": chunk.conversation_id,
                            "message_id": chunk.message_id
                        }))
                
                elif message_obj.type == "ping":
                    # Handle ping
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                else:
                    # Echo unknown message types
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "original": message_obj.dict()
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    except WebSocketDisconnect:
        connection_manager.disconnect(connection_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(connection_id, user_id)


# Tool endpoints
@app.get("/tools/list")
async def list_tools(current_user: Dict[str, Any] = Depends(get_read_user)):
    """List all available external tools"""
    try:
        tools = tool_registry.list_tools()
        return {"tools": tools, "count": len(tools)}
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/execute")
async def execute_tool(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Execute an external tool"""
    try:
        tool_name = request.get("tool_name")
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")
        
        # Extract parameters
        params = {k: v for k, v in request.items() if k != "tool_name"}
        
        # Execute tool
        result = await tool_registry.execute_tool(tool_name, **params)
        
        return {
            "tool_name": tool_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.post("/tools/web_search")
async def web_search(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Web search using Brave Search API"""
    try:
        query = request.get("query", "")
        count = request.get("count", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        
        result = await tool_registry.execute_tool("web_search", query=query, count=count)
        return result
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/weather")
async def get_weather(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Get weather information"""
    try:
        location = request.get("location", "")
        units = request.get("units", "metric")
        
        if not location:
            raise HTTPException(status_code=400, detail="location is required")
        
        result = await tool_registry.execute_tool("weather", location=location, units=units)
        return result
        
    except Exception as e:
        logger.error(f"Weather request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional utility endpoints
@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """Get messages for a conversation"""
    try:
        messages = await db_manager.get_conversation_messages(
            conversation_id,
            limit=limit,
            offset=offset
        )
        
        return {"messages": messages, "conversation_id": conversation_id}
        
    except Exception as e:
        logger.error(f"Failed to get conversation messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_read_user)
):
    """List conversations"""
    try:
        conversations = await db_manager.get_conversations(limit=limit, offset=offset)
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations")
async def create_conversation(
    conversation_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_write_user)
):
    """Create a new conversation"""
    try:
        conversation_id = await db_manager.create_conversation(
            conversation_data.get("conversation_id"),
            conversation_data.get("title"),
            conversation_data.get("metadata", {})
        )
        
        return {"conversation_id": conversation_id, "success": True}
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/mcp_server.log", rotation="10 MB", retention="7 days")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )