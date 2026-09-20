from functools import cache

from models.embedder import Embedder
from pipeline.retrieve import Retriever
from pipeline.vector_db import load_milvus_vector_store, load_neo4j_vector_store


@cache
def _get_retriever() -> Retriever:
    embedder = Embedder(embed_model="intfloat/multilingual-e5-small")
    # vector_store = load_milvus_vector_store(
    #     db_path="http://localhost:19530",
    #     collection_name="HID_docs"
    # )
    vector_store = load_neo4j_vector_store(
        url="bolt://localhost:7687",
        username="neo4j",
        password="neo4jneo4j"
    )
    return Retriever(
        embed_model=embedder,
        vector_store=vector_store
    )


def run_pipeline(query: str, top_k: int = 5) -> list[str]:
    return _get_retriever().retrieve_relevant_chunks(query=query, top_k=top_k)
