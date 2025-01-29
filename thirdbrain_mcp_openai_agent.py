from __future__ import annotations as _annotations

import httpx
from typing import Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from exceptions import ConfigurationError, ToolError, DatabaseConnectionError

import logging

logger = logging.getLogger(__name__)

# mcp client for pydantic ai
from mcp_client import MCPClient, Deps, logging, agent_loop

def validate_env_vars(required_vars: list[str]) -> None:
    """Validate that all required environment variables are set."""
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise ConfigurationError(f"Missing environment variables: {', '.join(missing_vars)}")

# Load environment variables
load_dotenv()

# Supabase setup
supabase: Client = None

# Define a context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting up the FastAPI application.")

    # Validate environment variables
    try:
        validate_env_vars(["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "API_BEARER_TOKEN"])
    except ConfigurationError as e:
        logger.error(f"Configuration error during startup: {e}")
        raise

    try:
        # Initialize Supabase client
        global supabase
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        if not supabase:
            logging.error("Supabase client is not initialized. Please check your environment variables.")
            raise DatabaseConnectionError("Supabase client initialization failed.")

        # Initialize MCPClient and connect to server
        global mcp_client
        mcp_client = MCPClient()
        await mcp_client.connect_to_server()
        logging.info("Startup tasks completed successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    # Yield control back to FastAPI
    yield

    # Shutdown logic
    logger.info("Shutting down the FastAPI application.")
    await mcp_client.cleanup()  
    logging.info("Shutdown tasks completed successfully.")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentResponse(BaseModel):
    success: bool

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

async def get_conversation_history(session_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return response.data[::-1]  # Reverse to get chronological order
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to fetch conversation history: {str(e)}")

async def save_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {"type": message_type, "content": content, **({"data": data} if data else {})}

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.get("/api/thirdbrain-hello")
async def thirdbrain_hello():
    return {"message": "Server is running"}

@app.post("/api/thirdbrain-mcp-openai-agent", response_model=AgentResponse)
async def thirdbrain_mcp_openai_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Fetch conversation history from the DB
        conversation_history = await get_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            #logger.debug("Processing message: %s", msg)
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]

            # Convert to appropriate message type for the agent
            if msg_type == "human":
                #messages.append(UserPromptPart(content=msg_content))
                messages.append({"role": "user", "content": msg_content})
            elif msg_type == "ai":
                #messages.append(TextPart(content=msg_content))
                messages.append({"role": "assistant", "content": msg_content})
            else:
                logging.debug("this was most likely an error message stored in the messages table")

        # Store user's query if it doesn't start with a slash
        if not request.query.startswith("/"):
            await save_message(
                session_id=request.session_id,
                message_type="human",
                content=request.query
            )

    except ToolError as e:
        logger.error(f"Tool error: {e}")
        raise HTTPException(status_code=500, detail=f"Tool error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during request processing")
        raise HTTPException(status_code=500, detail="Internal server error")
    

    # Get available tools and prepare them for the LLM
    tools = await mcp_client.get_available_tools()
    
    # Initialize agent dependencies
    async with httpx.AsyncClient() as client: 
        try:
            deps = Deps(
                client=client,
                supabase=supabase,
                session_id=request.session_id,
            )
            if request.query.startswith("/"):
                result = await mcp_client.handle_slash_commands(request.query)
            else:     
                result, messages = await agent_loop(request.query, tools, messages, deps)
            if request.query.startswith("/"):
                # Prepend the result with the slash command and server name
                command_info = f"Executed command: {request.query.split()[0]} {request.query.split()[1] if len(request.query.split()) > 1 else ''}".strip()
                result = f"{command_info}\n{result}"
            logging.info(f"Result: {result}")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting...")
            return
        except Exception as e:
            logging.error(f"Error in agent loop: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

        try:
            # Store agent's response
            await save_message(
                session_id=request.session_id,
                message_type="ai",
                content=result,
                data={"request_id": request.request_id}
            )

            return AgentResponse(success=True)
        except Exception as e:
            # Store error message in conversation
            await save_message(
                session_id=request.session_id,
                message_type="ai",
                content="I apologize, but I encountered an error processing your request.",
                data={"error": str(e), "request_id": request.request_id}
            )
            return AgentResponse(success=False)

if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)
