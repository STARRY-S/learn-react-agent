"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import os
from typing import Dict, List, Literal, cast
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient
from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


@asynccontextmanager
async def make_mcp_client_tools():
    """Generate Kubernetes MCP Client Tools"""
    server_url = os.getenv("MCP_SERVER")
    if server_url == "":
        server_url = "https://127.0.0.1:11434/sse"
    async with MultiServerMCPClient(
        {
            "kubernetes": {
                "url": server_url,
                "transport": "sse",
            }
        }
    ) as client:
        yield client.get_tools()


# Define the function that calls the model
async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    async with make_mcp_client_tools() as mcp_tools:
        all_tools = mcp_tools + TOOLS
        # Initialize the model with tool binding
        model = load_chat_model(configuration.model).bind_tools(all_tools)

        # Format the system prompt. Customize this to change the agent's behavior.
        system_message = configuration.system_prompt

        # Get the model's response
        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": system_message}, *state.messages]
            ),
        )
        # Return the model's response as a list to be added to existing messages
        return {"messages": [response]}

    return {}


# Create a tool handler node
async def tool_handler(state, config):
    """Process tool calls from the model.

    Args:
        state: The current state of the conversation.
        config: Configuration for the tool run.

    Returns:
        The result of processing the tools.
    """

    async with make_mcp_client_tools() as mcp_tools:
        all_tools = mcp_tools + TOOLS
        tool_node = ToolNode(all_tools)
        return await tool_node.ainvoke(state, config)

    tool_node = ToolNode(TOOLS)
    return await tool_node.ainvoke(state, config)


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", tool_handler)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
