from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_nomic.embeddings import NomicEmbeddings

# Load the Embeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", dimensionality=768)

# Neo4j graph credentials - Local Graph DB
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "rag_vector_2024")
database = "mjdb"

# Typical RAG retriever

typical_rag = Neo4jVector.from_existing_index(
    embedding = embeddings,
    url = URI, username=AUTH[0], password=AUTH[1], database=database,
    index_name="typical_rag",
)

# Parent retriever

parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
"""

parent_vectorstore = Neo4jVector.from_existing_index(
    embedding = embeddings,
    url = URI, username=AUTH[0], password=AUTH[1], database=database,
    index_name="parent_document",
    retrieval_query=parent_query,
)

# Hypothetic questions retriever

hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
    embedding = embeddings, 
    url = URI, username=AUTH[0], password=AUTH[1], database=database,
    index_name="hypothetical_questions",
    retrieval_query=hypothetic_question_query,
)
# Summary retriever

summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

summary_vectorstore = Neo4jVector.from_existing_index(
    embedding = embeddings, 
    url = URI, username=AUTH[0], password=AUTH[1], database=database,
    index_name="summary",
    retrieval_query=summary_query,
)
