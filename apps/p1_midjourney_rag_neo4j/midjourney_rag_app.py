"""
Simple demo of integration with ChainLit and LangGraph.
"""

import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from chromadb.config import Settings

from pprint import pprint
from typing import List, Annotated
from typing_extensions import TypedDict

import chainlit as cl



# Setup the Retriver

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

try:
    vectorstore = Chroma(persist_directory="./chroma_db", 
                         embedding_function=embedding_function,
                         collection_name='rag-chroma',
                         client_settings= Settings(anonymized_telemetry=False))
except:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function,
        persist_directory="./chroma_db",
        client_settings= Settings(anonymized_telemetry=False)
    )

    vectorstore.persist()
finally:
    retriever = vectorstore.as_retriever()
    print("Retriever is setup")


# Load llama model

local_llm = "llama3"

llm = ChatOllama(model=local_llm, 
                #  format="json", 
                 temperature=0)

print("Loaded Ollama Model")


##------------------------- Setup Agents and their Prompts ----------------------##

### Reformulate question

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You're an expert in question refiner.
    Given a chat history and the latest user question, you can either return the same question or formulate a standalone question based on following criteria,
    1) If the user question could not be understood without chat history, you should formulate a standalone question. It should include required context also
    2) If the user question does not have any reference to chat history, return the question as is. No modification required
    
    Your resonse should be in string format and do not include any preamble statement.

    Here is the chat hisotry between user and assistant till now for your reference:
    {chat_history}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {raw_question} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=['question', 'chat_history']
)

reformulate_question_chain = prompt | llm | StrOutputParser()


### Router

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()


### Retrieval Grader

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()


### Generate

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise. Provide the answer in string format. No key value pairs should be included in the answer
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

rag_chain = prompt | llm | StrOutputParser()


### Hallucination Grader

prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()


### Answer Grader

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()


### Search

# from langchain_community.tools.tavily_search import TavilySearchResults
# web_search_tool = TavilySearchResults(k=3)

from langchain_community.tools import DuckDuckGoSearchRun
web_search_tool = DuckDuckGoSearchRun()

# from langchain_community.tools import DuckDuckGoSearchResults
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
# web_search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

# Setup control flow in LangGraph

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: reformulated question question
        raw_question: raw question question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        messages: conversation history
    """

    question: str
    raw_question : str
    generation: str
    web_search: str
    documents: List[str]
    messages : Annotated[List, add_messages]


### Nodes


async def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = await retriever.ainvoke(question)
    return {"documents": documents, "question": question}


async def query_refiner(state):
    """
    Reformulate the question based on chat history and query

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, question
    """
    print("---QUERY REFINER---")
    raw_question = state["raw_question"]
    chat_history = state["messages"]

    # RAG generation
    refined_query = await reformulate_question_chain.ainvoke({"raw_question": raw_question, "chat_history":chat_history})
    return {"question": refined_query}

async def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = await rag_chain.ainvoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


async def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = await retrieval_grader.ainvoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


async def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = await web_search_tool.ainvoke({"query": question})
    # web_results = "\n".join(docs)
    web_results = Document(page_content=docs)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


### Conditional edge


async def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = await question_router.ainvoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


async def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


async def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = await hallucination_grader.ainvoke(
        {"documents": documents, "generation": generation}
    )

    # print(score)

    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = await answer_grader.ainvoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    

    
async def update_chat(state):
    """
    Update the conversation history in the state messages.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended messages to state dict
    """
    
    question = state["question"]
    raw_question = state["raw_question"]
    generation = state["generation"]

    return {'messages':[HumanMessage(content = raw_question),
                        AIMessage(content = f"Here is the refined question : \n{question}"),
                        AIMessage(content = generation)]}


graph = StateGraph(GraphState)

# Define the nodes
graph.add_node("query_refiner", query_refiner)  # web search
graph.add_node("websearch", web_search)  # web search
graph.add_node("retrieve", retrieve)  # retrieve
graph.add_node("grade_documents", grade_documents)  # grade documents
graph.add_node("generate", generate)  # generate
graph.add_node("update_chat", update_chat)  # update chat

# Build graph
graph.set_entry_point("query_refiner")

graph.add_conditional_edges(
    "query_refiner",
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
graph.add_edge("websearch", "generate")
graph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "update_chat",
        "not useful": "websearch",
    },
)

graph.add_edge("update_chat", END)


@cl.on_chat_start
async def on_chat_start():
    # save graph and state to the user session
    cl.user_session.set("graph", graph.compile())


import asyncio


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the graph and state from the user session
    graph: Runnable = cl.user_session.get("graph")

    # Stream the response to the UI
    ui_message = cl.Message(content="")

    config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    inputs = {"raw_question": message.content}

    async for output in graph.astream(inputs, config=config):
        for key, value in output.items():
            if key == "update_chat":
                content = value['messages'][-1].content


    chunk_size = 10
    for i in range(0, len(content), chunk_size):
        await ui_message.stream_token(token=content[i:i+chunk_size])
        await asyncio.sleep(0.02)

    await ui_message.send()