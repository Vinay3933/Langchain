import os
from typing import List

# from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import TokenTextSplitter
from neo4j.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser


# Neo4j graph credentials - Local Graph DB
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "rag_vector_2024")
database = "mjdb"


# dir_path = os.path.dirname(os.path.abspath("__file__"))
dir_path = "/Users/vinaytanwer/Desktop/Projects/Chatbots/langchain/apps/mj-app/packages/neo4j-advanced-rag"
mj_docs_dir = os.path.join(dir_path, "mj_scraper/mj_docs/final")

# File names to process
mj_files = sorted([file for file in os.listdir(mj_docs_dir) if file.endswith('.txt')])


# Load the Embeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", dimensionality=768)
embedding_dimension = 768

# Load the Chat Model
# llm = ChatOllama(model="llama3", temperature=0) #  format="json", 
llm = OllamaFunctions(model="llama3", temperature=0)

# Connect to Graph
graph = Neo4jGraph(URI, username=AUTH[0], password=AUTH[1], database=database)


# Define the Parent Child Text Splitters
parent_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=300)
child_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)



# Define Question Generation Retrieval chain
class Questions(BaseModel):
    """Generating hypothetical questions about text."""

    questions: List[str] = Field(
        ...,
        description=(
            "Generated hypothetical questions based on " "the information from the text"
        ),
    )

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are generating hypothetical questions based on the information "
                "found in the text. Make sure to provide full context in the generated "
                "questions."
            ),
        ),
        (
            "human",
            (
                "Use the given format to generate hypothetical questions from the "
                "following input: {input}"
            ),
        ),
    ]
)

question_chain = questions_prompt | llm.with_structured_output(Questions)

# question_chain.invoke({"input":text})


# Define Summary Generation Retrieval Chain
summary_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are expert in Summary writing.
        Your task is to generate accurate summaries for the input provided by the user
        Please respond in string format which can be parsed by a String parser
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Summary : 
        """,
        input_variables=["input"],
    )


summary_chain = summary_prompt | llm | StrOutputParser()

# summary_chain.invoke({"input":text})

# Process the files and ingest them into Neo4j Graph DB
for file_name in mj_files:

    # Check if the file has already been processed
    result = graph.query(
        "MATCH (p:Parent {file_name: $file_name}) RETURN p LIMIT 1",
        {"file_name": file_name},
    )
    if result:
        print(f"File {file_name} has already been processed. Skipping.")
        continue

    txt_path = os.path.join(mj_docs_dir, file_name)

    # Load the text file
    loader = TextLoader(str(txt_path))
    documents = loader.load()

    # Ingest Parent-Child node pairs
    parent_documents = parent_splitter.split_documents(documents)

    previous_parent_id = None

    for i, parent in enumerate(parent_documents):

        parent_id = f"{file_name}_{i}"

        child_documents = child_splitter.split_documents([parent])
        params = {
            "parent_text": parent.page_content,
            "parent_id": parent_id,
            "parent_embedding": embeddings.embed_query(parent.page_content),
            "children": [
                {
                    "text": c.page_content,
                    "id": f"{file_name}_{i}-{ic}",
                    "embedding": embeddings.embed_query(c.page_content),
                }
                for ic, c in enumerate(child_documents)
            ],
            "file_name" : file_name
        }
        # Ingest data
        graph.query(
            """
        MERGE (p:Parent {id: $parent_id})
        SET p.text = $parent_text,
            p.embedding = $parent_embedding,
            p.file_name = $file_name
        WITH p 
        UNWIND $children AS child
        MERGE (c:Child {id: child.id})
        SET c.text = child.text,
            c.embedding = child.embedding
        MERGE (c)<-[:HAS_CHILD]-(p)
        RETURN count(*)
        """,
            params,
        )

        # Create edges between consecutive parent nodes
        if previous_parent_id:
            graph.query(
                """
                MATCH (p1:Parent {id: $previous_parent_id}), (p2:Parent {id: $current_parent_id})
                MERGE (p1)-[:SAME_FILE_NEXT_SPLIT]->(p2)
                """,
                {"previous_parent_id": previous_parent_id, "current_parent_id": parent_id},
            )
        
        # Update the previous parent id for the next iteration
        previous_parent_id = parent_id


        # Create vector index for child
        try:
            graph.query(
                """
                CREATE VECTOR INDEX parent_document IF NOT EXISTS
                FOR (c:Child)
                ON c.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            pass

        # Create vector index for parents
        try:
            graph.query(
                """
                CREATE VECTOR INDEX typical_rag IF NOT EXISTS
                FOR (p:Parent)
                ON p.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            pass


    # Ingest hypothethical questions
    for i, parent in enumerate(parent_documents):
        questions = question_chain.invoke({"input": parent.page_content}).questions
        params = {
            "parent_id": f"{file_name}_{i}",
            "questions": [
                {"text": q, "id": f"{file_name}_{i}-{iq}", "embedding": embeddings.embed_query(q)}
                for iq, q in enumerate(questions)
                if q
            ],
        }
        graph.query(
            """
        MERGE (p:Parent {id: $parent_id})
        WITH p
        UNWIND $questions AS question
        CREATE (q:Question {id: question.id})
        SET q.text = question.text,
            q.embedding = question.embedding
        MERGE (q)<-[:HAS_QUESTION]-(p)
        RETURN count(*)
        """,
            params,
        )
        # Create vector index
        try:
            graph.query(
                """
                CREATE VECTOR INDEX hypothetical_questions IF NOT EXISTS
                FOR (q:Question)
                ON q.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            pass

    # Ingest summaries
    for i, parent in enumerate(parent_documents):
        summary = summary_chain.invoke({"input": parent.page_content})
        params = {
            "parent_id": f"{file_name}_{i}",
            "summary": summary,
            "embedding": embeddings.embed_query(summary),
        }
        graph.query(
            """
        MERGE (p:Parent {id: $parent_id})
        MERGE (p)-[:HAS_SUMMARY]->(s:Summary)
        SET s.text = $summary,
            s.embedding = $embedding
        RETURN count(*)
        """,
            params,
        )
        # Create vector index
        try:
            graph.query(
                """
                CREATE VECTOR INDEX summary IF NOT EXISTS
                FOR (s:Summary)
                ON s.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            pass

    print(f"File {file_name} processed successfully.")
