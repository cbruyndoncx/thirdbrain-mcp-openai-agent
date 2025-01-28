from __future__ import annotations as _annotations

import httpx
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Form
from contextlib import asynccontextmanager
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os

# mcp client for pydantic ai
from mcp_client import MCPClient, Deps, logging, agent_loop

# Load environment variables
load_dotenv()

# Supabase setup
supabase: Client = None

# Define a context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logging.debug("Starting up the FastAPI application...")

    # Check environment variables
    required_env_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "API_BEARER_TOKEN"]
    for var in required_env_vars:
        if not os.getenv(var):
            logging.error(f"Environment variable {var} is not set.")
            raise RuntimeError(f"Environment variable {var} is required but not set.")

    # Initialize Supabase client
    global supabase
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    if not supabase:
        logging.error("Supabase client is not initialized. Please check your environment variables.")
        raise RuntimeError("Supabase client initialization failed.")

    # Initialize MCPClient and connect to server
    global mcp_client
    mcp_client = MCPClient()
    await mcp_client.connect_to_server()
    logging.info("Startup tasks completed successfully.")

    # Yield control back to FastAPI
    yield

    # Shutdown logic
    logging.debug("Shutting down the FastAPI application...")
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

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

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
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            logging.debug("msg: ", msg)
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

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        ) 

    except Exception as e:
        logging.error(f"Error processing request - part 1: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

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
                result, messages = await agent_loop(request.query, tools, messages)
            logging.info(f"Result: {result}")
                
        except KeyboardInterrupt:
            logging.debug("Keyboard interrupt detected.")
            logging.debug("Exiting...")
            return
        except Exception as e:
            logging.error(f"Error in agent loop: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processung query for MCP request: {str(e)}")

        try:
            # Store agent's response
            await store_message(
                session_id=request.session_id,
                message_type="ai",
                content=result,
                data={"request_id": request.request_id}
            )

            return AgentResponse(success=True)
        except Exception as e:
            # Store error message in conversation
            await store_message(
                session_id=request.session_id,
                message_type="ai",
                content="I apologize, but I encountered an error processing your request.",
                data={"error": str(e), "request_id": request.request_id}
            )
            return AgentResponse(success=False)

if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ThirdBrain MCP OpenAI Agent.")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    # Set the logging level based on the argument
    logging_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging_level)
    logging.getLogger("mcp_client").setLevel(logging_level)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)
