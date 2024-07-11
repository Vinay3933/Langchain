import subprocess
import threading
import requests
import chainlit as cl
import asyncio
from langchain_core.runnables import Runnable, RunnableConfig


# Function to start the FastAPI server
def start_fastapi():
    subprocess.run(["python", "-m", "app.server"])

# Function to interact with the FastAPI server
def interact_with_api(prompt):
    response = requests.post("http://localhost:8000/mj-rag/invoke", 
                             json={'input':{"question": prompt}})
    return response.json()

# Start the server
fastapi_thread = threading.Thread(target=start_fastapi)
fastapi_thread.start()

@cl.on_chat_start
async def on_chat_start():
    # Start FastAPI server in a separate thread
    pass
    

@cl.on_message
async def on_message(message: cl.Message):

    # Stream the response to the UI
    ui_message = cl.Message(content="")
    
    reply = interact_with_api(message.content)
    print(reply)

    content = reply['output']

    chunk_size = 10
    for i in range(0, len(content), chunk_size):
        await ui_message.stream_token(token=content[i:i+chunk_size])
        await asyncio.sleep(0.02)

    await ui_message.send()