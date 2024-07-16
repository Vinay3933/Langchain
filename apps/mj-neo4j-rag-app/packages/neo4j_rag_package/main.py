from neo4j_advanced_rag.chain import chain

if __name__ == "__main__":
    original_query = "What is chaos in Midjourney?"
    print(
        chain.invoke(
            {"question": original_query,
            "configurable": {"strategy": "parent_strategy"}},
        )
    )
