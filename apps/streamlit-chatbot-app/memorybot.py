# Import necessary libraries
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField, RunnableParallel, ConfigurableFieldSpec
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime as dt
import json

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = {}
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = "Convo: " + dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")
if 'download_str' not in st.session_state:
        st.session_state.download_str = []

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.conversation_id = "Convo: " + dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")

# Display chat on UI
def display_chat():
    st.session_state.download_str = []
    for i in range(len(st.session_state['generated'])):
        st.chat_message("user").write(st.session_state["past"][i])
        st.chat_message("assistant").write(st.session_state["generated"][i])

# # Set up sidebar with various options
# with st.sidebar.expander("🛠️ ", expanded=False):
#     # Option to preview memory store
#     if st.checkbox("Preview memory store"):
#             f"{st.session_state.entity_memory}"


# Define Model to load and cache it

def load_converasation_runnable():

    # print("Inside LCR")

    print(st.session_state.MODEL, " ", st.session_state.temperature)
    llm = ChatOllama(model=st.session_state.MODEL, temperature=st.session_state.temperature)    
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

    runnable = prompt | llm | StrOutputParser()

    # Create a ConversationEntityMemory object if not already created
    def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
            if (user_id, conversation_id) not in st.session_state.entity_memory:
                st.session_state.entity_memory[(user_id, conversation_id)] = ChatMessageHistory()
            return st.session_state.entity_memory[(user_id, conversation_id)]


    # Create the ConversationChain object with the specified configuration
    st.session_state.Conversation_runnable = RunnableWithMessageHistory(
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


with st.sidebar.form("user_form"):
    user_id = st.text_input(label="User ID", key="user_id", value="Vinay")
    MODEL = st.selectbox(label='Model', options=['llama3','llama3:70b'], key="MODEL")
    temperature = st.slider("Model temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.1, key='temperature')
    # print("inside form")
    st.text_area("Model details", 
                 f"You are currently talking to {MODEL} model with temperature set at {temperature}",
                 disabled=True)
    st.form_submit_button("Update", on_click=load_converasation_runnable)
    # K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
st.title("🤖 Chat Bot with 🧠")
st.subheader(" Powered by 🦜 LangChain + Llama3 + Streamlit")

# Display chat_conversation
display_chat()

# Load the Conversation Runnable
if "Conversation_runnable" not in st.session_state:
     load_converasation_runnable()
Conversation_runnable = st.session_state.Conversation_runnable


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# # Get the user input
# user_input = get_text()

# # Generate the output using the ConversationChain object and the user input, and add the input/output to the session
# if user_input:
#     config = {"configurable": {"user_id": user_id, "conversation_id": st.session_state.conversation_id}}
#     output = Conversation.invoke({"input":user_input}, config=config)  
#     st.session_state.past.append(user_input)  
#     st.session_state.generated.append(output)

if user_input := st.chat_input():
    
    st.chat_message("user").write(user_input)

    config = {"configurable": {"user_id": user_id, "conversation_id": st.session_state.conversation_id}}
    
    output = st.chat_message("assistant").write_stream(Conversation_runnable.stream({"input":user_input}, config=config))

    # Append chat to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    # Append chat to downloaded str
    st.session_state.download_str.append("User: " + user_input)
    st.session_state.download_str.append("Assistant: " + output)
    st.session_state.download_str = '\n'.join(st.session_state.download_str)

# Allow to download as well
if st.session_state.download_str:
    st.sidebar.download_button('Download conversation',st.session_state.download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.button("Clear-all"):
        st.session_state["stored_session"] = []
        st.rerun()
