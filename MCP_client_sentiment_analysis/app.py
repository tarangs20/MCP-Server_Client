import gradio as gr
import os

from dotenv import load_dotenv

# This tells Python to find your .env file and load its contents!
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient


try:

    mcp_client = MCPClient(
        {"url": "https://tarangs20-mcp-sentiment-analysis.hf.space/gradio_api/mcp/sse", "transport": "sse",} # This is the MCP Client we created in the previous section
    )
    tools = mcp_client.get_tools()

    #for t in tools:
        #print("Name: ", t.name," description: ",t.description)

    model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
    agent = CodeAgent(tools=[*tools], model=model)

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        #type="messages",
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions."
    )

    demo.launch()
finally:
    mcp_client.disconnect()