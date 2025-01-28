# Base Sample Python Agent

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

This is a sample Python FastAPI agent that demonstrates the minimal required components to build an agent for the Live Agent Studio. It serves as a template and reference implementation for creating new Python-based agents.

This documentation is long but only because it gives you EVERYTHING you need to build a Python based agent for the Live Agent Studio! If you spend 5 minutes reading this carefully, you'll be ready to go.

## Overview;;

This agent provides a foundation for building AI-powered agents that can:
- Process natural language queries
- Maintain conversation history
- Integrate with external AI models
- Store and retrieve conversation data
- Handle authentication and security

**Supabase Version** (`sample_supabase_agent.py`): Uses Supabase client for database operations

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Supabase account
- Basic understanding of:
  - FastAPI and async Python
  - RESTful APIs
  - Pydantic models
  - Environment variables

## Core Components

### 1. FastAPI Application (`sample_supabase_agent.py`)

The main application is built using FastAPI, providing:

- **Authentication**
  - Bearer token validation via environment variables
  - Secure endpoint protection
  - Customizable security middleware

- **Request Handling**
  - Async endpoint processing
  - Structured request validation
  - Error handling and HTTP status codes

- **Database Integration**
  - Supabase connection management
  - Message storage and retrieval
  - Session-based conversation tracking

### 2. Data Models

#### Request Model
```python
class AgentRequest(BaseModel):
    query: str        # The user's input text
    user_id: str      # Unique identifier for the user
    request_id: str   # Unique identifier for this request
    session_id: str   # Current conversation session ID
```

#### Response Model
```python
class AgentResponse(BaseModel):
    success: bool     # Indicates if the request was processed successfully
```

### 3. Database Schema

The agent uses Supabase tables with the following structure:

#### Messages Table
```sql
messages (
    id: uuid primary key
    created_at: timestamp with time zone
    session_id: text
    message: jsonb {
        type: string       # 'human' or 'assistant'
        content: string    # The message content
        data: jsonb       # Optional additional data
    }
)
```

## Setup

   # Copy example environment file
   cp .env.example .env

   # Edit .env with your credentials
   nano .env  # or use your preferred editor

2. **Configure Environment Variables**

   > ⚠️ **Important**: For Docker, do not wrap environment variable values in quotes, even if they contain special characters. Docker will handle the values correctly without quotes.

   #### Supabase Configuration
   Required environment variables in `.env` file (do not use quotes around values):
   ```plaintext
   SUPABASE_URL=your-project-url
   SUPABASE_SERVICE_KEY=your-service-key
   API_BEARER_TOKEN=your-token-here
   ```

3. **Create Database Tables**
   You'll need to create the following tables:
   ```sql
   -- Enable the pgcrypto extension for UUID generation
   CREATE EXTENSION IF NOT EXISTS pgcrypto;

   CREATE TABLE messages (
       id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       session_id TEXT NOT NULL,
       message JSONB NOT NULL
   );

   CREATE INDEX idx_messages_session_id ON messages(session_id);
   CREATE INDEX idx_messages_created_at ON messages(created_at);
   ```

   > Note: If you're using Supabase, the `pgcrypto` extension is already enabled by default.

## Installation Methods

### Docker Installation (Recommended)

1. Build the base image first (make sure Docker is running on your machine):
```bash
cd ../base_python_docker
docker build -t ottomator/base-python:latest .
cd ../~sample-python-agent~
```

2. Build the agent image (you can swap between Supabase and PostgreSQL versions in the `Dockerfile`):
```bash
docker build -t sample-python-agent .
```

3. Run the container:
```bash
docker run -d --name sample-python-agent -p 8001:8001 --env-file .env sample-python-agent
```

The agent will be available at `http://localhost:8001`.

### Local Installation (Docker Alternative)

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Run the agent:

```bash
uvicorn sample_supabase_agent:app --host 0.0.0.0 --port 8001
```

## Configuration

The agent uses environment variables for configuration. 
You can set these variables in a `.env` file or using your operating system's environment variable settings.

## Making Your First Request

Test your agent using curl or any HTTP client:

### Supabase Version
```bash
curl -X POST http://localhost:8001/api/sample-supabase-agent \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hello, agent!",
    "user_id": "test-user",
    "request_id": "test-request-1",
    "session_id": "test-session-1"
  }'
```

## Building Your Own Agent

1. **Fork the Template**
   - Create a copy of this sample agent
   - Rename files and update paths as needed

2. **Customize the Agent Logic**
   - Locate the TODO section in `sample_supabase_agent.py`
   - Add your custom agent logic:
     ```python
     # Example: Add AI model integration
     async def get_ai_response(query: str, history: List[Dict]) -> str:
         # Add your AI model logic here
         # e.g., OpenAI, Anthropic, or custom models
         return "AI response"

     @app.post("/api/your-agent", response_model=AgentResponse)
     async def your_agent(request: AgentRequest):
         # Get conversation history
         history = await fetch_conversation_history(request.session_id)
         
         # Process with your AI model
         response = await get_ai_response(request.query, history)
         
         # Store the response
         await store_message(
             session_id=request.session_id,
             message_type="assistant",
             content=response
         )
         
         return AgentResponse(success=True)
     ```

3. **Add Dependencies**
   - Update `requirements.txt` with new packages
   - Document any external services or APIs

4. **Implement Error Handling**
   ```python
   try:
       result = await your_operation()
   except YourCustomError as e:
       raise HTTPException(
           status_code=400,
           detail=f"Operation failed: {str(e)}"
       )
   ```

## Troubleshooting - Common issues and solutions:

1. **Authentication Errors**
   - Verify bearer token in environment
   - Check Authorization header format
   - Ensure token matches exactly

2. **Database Connection Issues**
   - For Supabase:
     - Verify Supabase credentials
     - Validate table permissions

3. **Performance Problems**
   - Check database query performance
   - Consider caching frequently accessed data
