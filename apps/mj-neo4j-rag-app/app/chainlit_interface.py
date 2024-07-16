import subprocess
import threading
import requests
import chainlit as cl
import asyncio
from langchain_core.runnables import Runnable, RunnableConfig
from chainlit.input_widget import Select, Switch, Slider


# Function to start the FastAPI server
def start_fastapi():
    subprocess.run(["python", "-m", "app.server"])

# Function to interact with the FastAPI server
def interact_with_api(prompt, retrieval_strategy):
    strategy = {'strategy': retrieval_strategy}
    print(strategy)
    response = requests.post("http://localhost:8000/mj-rag/invoke", 
                             json={'input':{"question": prompt,
                                            "configurable": retrieval_strategy}})
    return response.json()

# Start the server
fastapi_thread = threading.Thread(target=start_fastapi)
fastapi_thread.start()

@cl.on_chat_start
async def on_chat_start():
    # Start FastAPI server in a separate thread

    settings = await cl.ChatSettings(
            [
                Select(
                    id="rag_strategy",
                    label="Retrieval Strategy",
                    values=["typical_rag", "parent_strategy", "hypothetical_questions", "summary_strategy"],
                    initial_index=0,
                ),
            ]
        ).send()
    
    await setup_agent(settings)

@cl.on_settings_update
async def setup_agent(settings):
    retrieval_strategy = settings['rag_strategy']

    cl.user_session.set("strategy", retrieval_strategy)
    

@cl.on_message
async def on_message(message: cl.Message):

    retrieval_strategy = cl.user_session.get("strategy")

    # Stream the response to the UI
    ui_message = cl.Message(content="")
    
    reply = interact_with_api(message.content, retrieval_strategy)
    print(reply)

    content = reply['output']

    chunk_size = 10
    for i in range(0, len(content), chunk_size):
        await ui_message.stream_token(token=content[i:i+chunk_size])
        await asyncio.sleep(0.02)

    await ui_message.send()