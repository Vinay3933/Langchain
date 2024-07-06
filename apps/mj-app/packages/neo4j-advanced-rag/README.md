# neo4j-advanced-rag

This template allows you to balance precise embeddings and context retention by implementing advanced retrieval strategies.

## Strategies

1. **Typical RAG**:
   - Traditional method where the exact data indexed is the data retrieved.
2. **Parent retriever**:
   - Instead of indexing entire documents, data is divided into smaller chunks, referred to as Parent and Child documents.
   - Child documents are indexed for better representation of specific concepts, while parent documents is retrieved to ensure context retention.
3. **Hypothetical Questions**:
   - Documents are processed to determine potential questions they might answer.
   - These questions are then indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
4. **Summaries**:
   - Instead of indexing the entire document, a summary of the document is created and indexed.
   - Similarly, the parent document is retrieved in a RAG application.
