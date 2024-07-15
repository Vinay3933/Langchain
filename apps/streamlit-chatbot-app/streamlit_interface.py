# import subprocess
# import threading
# import requests
import chainlit as cl
import asyncio
from langchain_core.runnables import Runnable, RunnableConfig


###################### Sandbox LLM Chain - functionality testing ######################


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField, RunnableParallel, ConfigurableFieldSpec
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from chainlit.input_widget import Select, Switch, Slider



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful chat assistant. Respond to user question in 20 words or fewer \
             and if you don't know the answer, say that you don't know.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def runnable_with_history(model_name="llama3",  temp=0, store={}):

    llm = ChatOllama(model=model_name, temperature=0)
    runnable = prompt | llm | StrOutputParser()
    
    def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in store:
            store[(user_id, conversation_id)] = ChatMessageHistory()
        return store[(user_id, conversation_id)]


    runnable_with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            ),
        ],
    )

    return runnable_with_message_history

#######################################################################################


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="llama3:8b",
            markdown_description="The underlying LLM model is **llama3:8b**.",
            icon="https://img.freepik.com/premium-vector/cute-business-llama-icon-illustration-alpaca-mascot-cartoon-character-animal-icon-concept-isolated_138676-989.jpg",
        ),
        cl.ChatProfile(
            name="llama3:70b",
            markdown_description="The underlying LLM model is **llama3:70b**.",
            icon="https://static.vecteezy.com/system/resources/thumbnails/001/874/042/small/cute-happiness-llama-free-vector.jpg",
        ),
    ]

# @cl.password_auth_callback
# def auth_callback(username: str, password: str) -> Optional[cl.User]:
#     if (username, password) == ("admin", "admin"):
#         return cl.User(identifier="admin", metadata={"role": "ADMIN"})
#     else:
#         return None


@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(content=f"Starting chat using the {chat_profile} chat profile").send()

    settings = await cl.ChatSettings(
        [
            # Select(
            #     id="Model",
            #     label="Meta Llama Model",
            #     values=["llama3:8b", "llama3:70b"],
            #     initial_index=0,
            # ),
            Slider(
                id="Temperature",
                label="Llama - Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()

    model = "llama:70b" if chat_profile == "llama:70b" else "llama3"


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    



    

    

# @cl.on_message
# async def on_message(message: cl.Message):

#     # Stream the response to the UI
#     ui_message = cl.Message(content="")
    
#     reply = interact_with_api(message.content)

#     content = reply['output']

#     chunk_size = 10
#     for i in range(0, len(content), chunk_size):
#         await ui_message.stream_token(token=content[i:i+chunk_size])
#         await asyncio.sleep(0.02)

#     await ui_message.send()