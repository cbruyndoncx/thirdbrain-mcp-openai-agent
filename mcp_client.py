import os
import asyncio
import json
import logging
import pprint

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Any
from contextlib import AsyncExitStack
from colorama import init, Fore, Style
init(autoreset=True)  # Initialize colorama with autoreset=True

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext 
from pydantic_ai.tools import Tool, ToolDefinition

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from httpx import AsyncClient
from supabase import Client

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel

# Get the logger used by uvicorn
logging = logging.getLogger("uvicorn")
logging.setLevel("INFO")

# Load environment variables from .env
load_dotenv()  
# Get the selected provider
selected = os.getenv("SELECTED")

# Check if SELECTED is defined
if not selected:
    raise ValueError("SELECTED is not defined in the .env file.")

# Resolve BASE_URL, API_KEY, and LLM_MODEL dynamically
base_url = os.getenv(f"{selected}_URL")
api_key = os.getenv(f"{selected}_API_KEY")
llm_model = os.getenv(f"{selected}_MODEL")

# Check if the resolved variables exist
if not base_url:
    raise ValueError(f"{selected}_URL is not defined in the .env file.")
if not api_key:
    raise ValueError(f"{selected}_API_KEY is not defined in the .env file.")
if not llm_model:
    raise ValueError(f"{selected}_MODEL is not defined in the .env file.")

# Print the resolved values
logging.debug(f"SELECTED: {selected}")
logging.debug(f"BASE_URL: {base_url}")
logging.debug(f"API_KEY: {api_key}")
logging.debug(f"LLM_MODEL: {llm_model}")

client = AsyncOpenAI( 
    base_url=base_url,
    api_key=api_key)
model = OpenAIModel(
    llm_model,
    base_url=base_url,
    api_key=api_key)

# System prompt that guides the LLM's behavior and capabilities
# This helps the model understand its role and available tools
SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat. Use the responses from these function calls to provide accurate and informative answers. The answers should be natural and hide the fact that you are using tools to access real-time information. Guide the user about available tools and their capabilities. Always utilize tools to access real-time information when required. Engage in a friendly manner to enhance the chat experience.
 
# Tools
 
{tools}
 
# Notes 
 
- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""
 
@dataclass
class Deps:
    client: AsyncClient
    supabase: Client
    session_id: str

class MCPClient:
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    This class manages the connection and communication with the tools through MCP.
    """
    def __init__(self):
        # Initialize sessions and agents dictionaries
        self.sessions: Dict[str, ClientSession] = {}  # Dictionary to store {server_name: session}
        self.agents: Dict[str, Agent] = {}  # Dictionary to store {server_name: agent}
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.tools = {}
        self.connected = False
        self.config_file = 'mcp_config.json'
        self.dynamic_tools: List[Tool] = []  # List to store dynamic pydantic tools

    async def connect_to_server(self):
        if self.connected:
            logging.info("Already connected to servers.")
            return

        logging.debug(f"Loading {self.config_file} ...")
        try:
            with open(self.config_file) as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{self.config_file} file not found.")
            return
        except json.JSONDecodeError:
            logging.error(f"{self.config_file} is not a valid JSON file.")
            return
        finally:
            f.close()  # Ensure the file is always closed    
        
        logging.debug("Available servers in config: ", list(config['mcpServers'].keys()))
        logging.debug("Full config content: ", json.dumps(config, indent=2))
        
        # Connect to all servers in config
        for server_name, server_config in config['mcpServers'].items():
            logging.debug(f"Attempting to load {server_name} server config...")
            logging.debug("Server config found:", json.dumps(server_config, indent=2))
            
            server_params = StdioServerParameters(
                command=server_config['command'],
                args=server_config['args'],
                env=server_config.get('env'),
            )
            logging.debug("Created server parameters:", server_params)
           
            try:
                # Create and store session with server name as key
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                self.sessions[server_name] = session
                
                # Create and store an Agent for this server
                server_agent: Agent = Agent(
                    model,
                    system_prompt=(
                        f"You are an AI assistant that helps interact with the {server_name} server. "
                        "You will use the available tools to process requests and provide responses."
                        "Make sure to always give feedback to the user after you have called the tool, especially when the tool does not generate any message itself."
                    )
                )
                self.agents[server_name] = server_agent
                
                # List available tools for this server
                response = await session.list_tools()
                server_tools = [{
                    "name": f"{server_name}__{tool.name}",
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]
            except Exception as e:
                error_message = f"Failed to connect to and get tools from server {server_name}: {str(e)}"
                logging.error(error_message)
                return error_message
            
            # Add server's tools to overall available tools
            self.available_tools.extend(server_tools)

            # Create corresponding dynamic pydantic tools
            for tool in response.tools:
                async def prepare_tool(
                    ctx: RunContext[str], 
                    tool_def: ToolDefinition,
                    tool_name: str = tool.name,
                    server: str = server_name
                ) -> Union[ToolDefinition, None]:
                    # Customize tool definition based on server context
                    tool_def.name = f"{server}__{tool_name}"
                    tool_def.description = f"Tool from {server} server: {tool.description}"
                    logging.info(tool_def.description)
                    return tool_def

                async def tool_func(ctx: RunContext[Any], str_arg) -> str:
                    agent_response = await server_agent.run_sync(str_arg)
                    logging.debug(f"Server agent response: {agent_response}")
                    logging.info(f"Tool {tool.name} called with {str_arg}. Agent response: {agent_response}")
                    return f"Tool {tool.name} called with {str_arg}. Agent response: {agent_response}"               
                
                # Long descriptions beyond 1023 are not supported with OpenAI,
                # so replacing with a local file description optimized for use if it exists.
                file_name = f"./mcp-tool-description-overrides/{server_name}__{tool.name}"

                if os.path.exists(file_name):
                    try:
                        with open(file_name, 'r') as f:
                            file_content = f.read()
                        tool.description = file_content
                    except Exception as e:
                        logging.error(f"An error occurred while reading the file: {e}")
                    finally: 
                        f.close
                else:
                    logging.debug(f"File '{file_name}' not found. Using default description")

                dynamic_tool = Tool(
                    tool_func,
                    prepare=prepare_tool,
                    name=f"{server_name}__{tool.name}",
                    description=tool.description
                )
                self.tools[tool.name] = {
                    "name": tool.name,
                    "callable": self.call_tool(f"{server_name}__{tool.name}"),
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    },
                }
                logging.debug(f"Added tool: {tool.name}")
            
            logging.info(f"Connected to server {server_name} with tools: %s", [tool["name"] for tool in server_tools])

            self.connected = True
        logging.info("Done connecting to servers.")

    async def add_mcp_configuration(self, query: str) -> Optional[str]:
        """
        Add a new MCP server configuration if the query starts with 'mcpServer'.
        The query should be in the format:
        {"server_name": {"command": "command", "args": ["arg1", "arg2"], "env": null}}
        """
        
        try:
            # Extract the JSON part of the query
            config_str = query
            new_config = json.loads(config_str)

            # Validate the new configuration
            if not isinstance(new_config, dict):
                return "Error: Configuration must be a JSON object."

            # The server name is the key in the new_config dictionary
            server_name = next(iter(new_config), None)
            if not server_name:
                return "Error: Server name is required as the key in the configuration."

            server_config = new_config[server_name]

            # Validate the server configuration
            if not isinstance(server_config, dict):
                return f"Error: Configuration for server '{server_name}' must be a JSON object."

            if "command" not in server_config or "args" not in server_config:
                return f"Error: 'command' and 'args' are required for server '{server_name}'."

            # Load the existing config
            try:
                with open("mcp_config.json", "r") as f:
                    config = json.load(f)
            except FileNotFoundError:
                return "Error: mcp_config.json file not found."
            except json.JSONDecodeError:
                return "Error: mcp_config.json is not a valid JSON file."

            # Check if the server name already exists
            if server_name in config.get("mcpServers", {}):
                return f"Error: Server '{server_name}' already exists in the configuration."

            # Add the new server configuration
            if "mcpServers" not in config:
                config["mcpServers"] = {}

            config["mcpServers"][server_name] = {
                "command": server_config["command"],
                "args": server_config["args"],
                "env": server_config.get("env")  # Optional field
            }

            # Save the updated config back to the file
            with open("mcp_config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Connect to the new server
            await self.connect_to_server_with_config(server_name, config["mcpServers"][server_name])

            return f"Successfully added and connected to server '{server_name}'."

        except json.JSONDecodeError:
            return "Error: Invalid JSON format in the query."
        except Exception as e:
            return f"Error adding MCP configuration: {str(e)}"
        return None

    async def connect_to_server_with_config(self, server_name: str, server_config: dict):
        """Connect to a server using the provided configuration."""
        server_params = StdioServerParameters(
            command=server_config['command'],
            args=server_config['args'],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        self.sessions[server_name] = session

        response = await session.list_tools()
        server_tools = [{
            "name": f"{server_name}__{tool.name}",
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        self.available_tools.extend(server_tools)
        return None

    async def list_mcp_servers(self) -> str:
        """List all MCP servers in the configuration."""
        try:
            with open("mcp_config.json", "r") as f:
                config = json.load(f)
            servers = config.get("mcpServers", {}).keys()
            return "MCP Servers: " + ", ".join(servers)
        except FileNotFoundError:
            return "Error: mcp_config.json file not found."
        except json.JSONDecodeError:
            return "Error: mcp_config.json is not a valid JSON file."

    async def list_server_functions(self, server_name: str) -> str:
        """List all functions provided by a specific MCP server."""
        if server_name not in self.sessions:
            return f"Error: Server '{server_name}' is not connected."
        try:
            response = await self.sessions[server_name].list_tools()
            functions = []
            for tool in response.tools:
                parameters = tool.inputSchema.get('properties', {})
                functions.append({
                    "name": tool.name,
                    "parameters": parameters
                })
            formatted_output = json.dumps(functions, indent=2)
            return f"Functions for server '{server_name}':\n```\n{formatted_output}\n```"
        except Exception as e:
            return f"Error listing functions for server '{server_name}': {str(e)}"

    async def cleanup(self):
        """Clean up resources."""
        logging.debug("Cleaning up resources...")
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.available_tools.clear()
        self.connected = False
        logging.info("Cleanup completed.")
    
    async def get_available_tools(self) -> List[Any]:
        """
        Retrieve a list of available tools from the MCP server.
        Simplify the schema for each tool to make it compatible with the OpenAI API.
        """
        if not self.sessions:
            raise RuntimeError("Not connected to MCP server")

    
        def simplify_schema(schema):
            """
            Simplifies a JSON schema by removing unsupported constructs like 'allOf', 'oneOf', etc.,
            and preserving the core structure and properties. Needed for pandoc to work with the LLM.

            Args:
                schema (dict): The original JSON schema.

            Returns:
                dict: A simplified JSON schema.
            """
            # Create a new schema with only the basic structure
            simplified_schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": schema.get("additionalProperties", False)
            }

            # Remove unsupported constructs like 'allOf', 'oneOf', 'anyOf', 'not', 'enum' at the top level
            for key in ["allOf", "oneOf", "anyOf", "not", "enum"]:
                if key in simplified_schema:
                    del simplified_schema[key]

            return simplified_schema

        return  {
            tool['name']: {
                "name": tool['name'],
                "callable": self.call_tool(
                    tool['name']
                ),  # returns a callable function for the rpc call
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool['description'][:1023],
                        "parameters": simplify_schema(tool['input_schema'])
                    },
                },
            }
            for tool in self.available_tools
            if tool['name']
            != "xxx"  # Excludes xxx tool as it has an incorrect schema
        }
        
    def call_tool(self, server__tool_name: str) -> Any:
        """
        Create a callable function for a specific tool.
        This allows us to execute functions through the MCP server.
 
        Args:
            tool_name: The name of the tool to create a callable for
 
        Returns:
            A callable async function that executes the specified tool
        """
        server_name, tool_name = server__tool_name.split("__")  

        if not server_name in self.sessions:
            raise RuntimeError("Not connected to MCP server")
 
        async def callable(*args, **kwargs):
            try:
                response = await asyncio.wait_for(
                    self.sessions[server_name].call_tool(tool_name, arguments=kwargs),
                    timeout=10.0  # Set a timeout
                )
                return response.content[0].text if response.content else None
            except asyncio.TimeoutError:
                logging.debug("Timeout while calling MCP server")
                return None
            except Exception as e:
                logging.error(f"Error calling MCP server: {e}")
                return None
 
        return callable
    
    async def drop_mcp_server(self, server_name: str) -> str:
        """Remove an MCP server from the configuration and disconnect it."""
        try:
            with open("mcp_config.json", "r") as f:
                config = json.load(f)

            if server_name not in config.get("mcpServers", {}):
                return f"Error: Server '{server_name}' does not exist in the configuration."

            # Remove the server from the configuration
            del config["mcpServers"][server_name]

            # Save the updated config back to the file
            with open("mcp_config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Disconnect the server if it is connected
            if server_name in self.sessions:
                await self.sessions[server_name].close()
                del self.sessions[server_name]
                del self.agents[server_name]

            return f"Successfully removed and disconnected server '{server_name}'."

        except FileNotFoundError:
            return "Error: mcp_config.json file not found."
        except json.JSONDecodeError:
            return "Error: mcp_config.json is not a valid JSON file."
        except Exception as e:
            return f"Error removing MCP server: {str(e)}"
        """
        Handle slash commands for adding MCP servers and listing available functions.
        """
        try:
            command, *args = query.split()
            if command == "/addMcpServer":
                result = await self.add_mcp_configuration(" ".join(args))
            elif command == "/list":
                result =  await self.list_mcp_servers()
            elif command == "/functions" and args:
                result =  await self.list_server_functions(args[0])
            elif command == "/dropMcpServer" and args:
                result = await self.drop_mcp_server(args[0])
                result =  "Error: Invalid command or missing arguments."
        except Exception as e:
            logging.error(f"Error handling slash commands: {e}")
            return None

        return result
    
async def agent_loop(query: str, tools: dict, messages: List[dict] = None):
    """
    Main interaction loop that processes user queries using the LLM and available tools.
 
    This function:
    1. Sends the user query to the LLM with context about available tools
    2. Processes the LLM's response, including any tool calls
    3. Returns the final response to the user
 
    Args:
        query: User's input question or command
        tools: Dictionary of available tools and their schemas
        messages: List of messages to pass to the LLM, defaults to None
    """
 
    messages = (
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools="\n- ".join(
                        [
                            f"{t['name']}: {t['schema']['function']['description']}"
                            for t in tools.values()
                        ]
                    )
                ),  # Creates System prompt based on available MCP server tools
            },
        ]
        if messages is None
        else messages  # reuse existing messages if provided
    )
    # add user query to the messages list
    messages.append({"role": "user", "content": query})
    pprint.pprint(messages)

    # Query LLM with the system prompt, user query, and available tools
    first_response = await client.chat.completions.create(
        model=llm_model,
        messages=messages,
        tools=([t["schema"] for t in tools.values()] if len(tools) > 0 else None),
        max_tokens=4096,
        temperature=0,
    )
    # detect how the LLM call was completed:
    # tool_calls: if the LLM used a tool
    # stop: If the LLM generated a general response, e.g. "Hello, how can I help you today?"
    stop_reason = (
        "tool_calls"
        if first_response.choices[0].message.tool_calls is not None
        else first_response.choices[0].finish_reason
    )
 
    if stop_reason == "tool_calls":
        # Extract tool use details from response
        for tool_call in first_response.choices[0].message.tool_calls:
            arguments = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )
            # Call the tool with the arguments using our callable initialized in the tools dict
            logging.debug(tool_call.function.name)
            tool_result = await tools[tool_call.function.name]["callable"](**arguments)
            if tool_result is None:
                tool_result = f"{tool_call.function.name}"
            logging.debug("tool result begin")
            pprint.pprint(tool_result)
            logging.debug("tool result end")

            # Add tool call to messages with an id
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,  # Include the tool_call_id here
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(arguments)
                    }
                }]
            })
            
            # Add the tool result to the messages list
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(tool_result),
                }
            )
            logging.debug("before new response all messages")
            pprint.pprint(messages)

        # Query LLM with the user query and the tool results
        new_response = await client.chat.completions.create(
            model=llm_model,
            messages=messages,
        )
 
    elif stop_reason == "stop":
        # If the LLM stopped on its own, use the first response
        new_response = first_response
    else:
        raise ValueError(f"Unknown stop reason: {stop_reason}")
    
    # Add the LLM response to the messages list
    messages.append(
        {"role": "assistant", "content": new_response.choices[0].message.content}
    )

    # Return the LLM response and messages
    return new_response.choices[0].message.content, messages


async def main():
    """
    Main function that sets up the MCP server, initializes tools, and runs the interactive loop.
    """
    mcp_client = MCPClient()
    await mcp_client.connect_to_server()

    tools = await mcp_client.get_available_tools()
    
    # Start interactive prompt loop for user queries
    messages = None
    while True:
        try:
            # Get user input and check for exit commands
            user_input = input("\nEnter your prompt (or 'quit' to exit): ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if user_input.startswith("/"):
                response = await mcp_client.handle_slash_commands(user_input)
            else:
                # Process the prompt and run agent loop
                response, messages = await agent_loop(user_input, tools, messages)
            logging.debug("Response:", response)
            # logging.debug("Messages:", messages)
        except KeyboardInterrupt:
            logging.debug("Exiting...")
            break
        except Exception as e:
            logging.error(f"Error occurred: {e}")
 
    await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
