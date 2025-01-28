# thirdbrain-mcp-openai-agent
mcp agent for openai created for the oTTomator hackathon.
This agent makes mcp tools accessible to openai compatible llms (chat completions)
The models that can be used are amongst others evidently OpenAI, but also deepseek-chat and ollama.
At this point deepseek-r1 does not support function calling.


# Specific ThirdBrain_mcp_agent

Add a new mcpServer by chat starting with ```/addMcpServer``` and **valid json** config, is saved in mcp_config.json

```/addMcpServer {"memory": {"command": "docker", "args": ["run", "-i", "--rm", "mcp/memory"]}}```

Remove an mcp server using```/dropMcpServer <name>``` 

# mcp-proxy-pydantic-agent

This agent is based upon the code in https://github.com/p2c2e/mcp_proxy_pydantic_agent
and the exellent wip found at https://github.com/philschmid/mcp-openai-gemini-llama-example

CAVEAT - it appears to be impossible to use the pydantic-ai agent system as parts of the tooling is async and other parts sync. 
Tried but did not find a solution.

And it is recognized as a bug still as of end of januari 2025
https://github.com/pydantic/pydantic-ai/issues/149

and is being worked on in major Agent rewrite as of 28 jan 25
https://github.com/pydantic/pydantic-ai/pull/725



# Slash commands

## /addMcpServer {servername: { command: X , args: [], env: {} }}
This command allows you to add a new MCP server configuration. The configuration should be in valid JSON format and will be saved in the `mcp_config.json` file. The `command` specifies the executable to run, `args` is an array of arguments for the command, and `env` is an optional object for environment variables.

Example:
/addMcpServer {"memory": {"command": "docker", "args": ["run", "-i", "--rm", "mcp/memory"]}}


## /disable [server-name, ...]
This command disables one or more specified MCP servers. Provide the server names as a comma-separated list.

Example:
/disable server1, server2

## /enable [server-name, ...]
This command enables one or more specified MCP servers. Provide the server names as a comma-separated list.

Example:
/enable server1, server2

## /dropMcpServer
This command removes an existing MCP server configuration. Provide the server name you wish to remove.

Example:
/dropMcpServer server1


# mcp_config.json default setup
The `mcp_config.json` file contains the default configurations for MCP servers. Each server configuration includes the command to execute, arguments, and optional environment variables. This file is essential for the MCP agent to know how to interact with different servers.

Example default setup:
```json
{
    "memory": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "mcp/memory"],
        "env": {}
    },
    "compute": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "mcp/compute"],
        "env": {}
    }
}