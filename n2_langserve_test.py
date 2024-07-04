
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser


from fastapi import FastAPI
from langserve import add_routes


# Load llama3 model using Ollama
model = Ollama(model="llama3")

# llm.invoke("Tell me a joke")

from langchain_core.prompts import ChatPromptTemplate

system_template = "You're an helpful assistant given a role to respond like a {role}."
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser



# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)