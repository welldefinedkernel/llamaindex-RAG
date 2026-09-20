from functools import cache

from models.embedder import Embedder
from pipeline.retrieve import Retriever
from pipeline.vector_db import load_vector_store


@cache
def _get_retriever() -> Retriever:
    embedder = Embedder(embed_model="microsoft/harrier-oss-v1-0.6b")
    vector_store = load_vector_store(
        db_path="http://localhost:19530",
        collection_name="HID_docs"
    )
    return Retriever(
        embed_model=embedder,
        vector_store=vector_store
    )


def run_pipeline(query: str, top_k: int = 5) -> list[str]:
    return _get_retriever().retrieve_relevant_chunks(query=query, top_k=top_k)
