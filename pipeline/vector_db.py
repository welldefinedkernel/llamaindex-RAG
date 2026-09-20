from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# Overwritten from the index at connect time; 1 makes a missing index fail loudly rather than silently.
_INDEX_INFERRED_DIM = 1

# Appended to the index lookup by Neo4jVectorStore; the text/score/id aliases are read by name.
# Fields are folded into text because node.metadata is dropped for graphs not ingested by LlamaIndex.
# Raw string so \n reaches Cypher as an escape rather than a literal newline inside the quotes.
_NEO4J_RETRIEVAL_QUERY = r"""
RETURN reduce(acc = '', line IN [
         f IN [
           'displayLabel: ' + toString(node.displayLabel),
           'kind: '         + toString(node.kind),
           'status: '       + toString(node.status),
           'summary: '      + toString(node.summary),
           'name: '         + toString(node.name)
         ] WHERE f IS NOT NULL
       ] | acc + line + '\n') + coalesce(node.content, node.description, '') AS text,
       score,
       coalesce(node.id, elementId(node)) AS id
"""


def create_vector_store(
    db_path: str,
    collection_name: str,
    embedding_dim: int,
    overwrite: bool = False,
) -> MilvusVectorStore:
    if not db_path.startswith("http://") and not db_path.startswith("https://"):
        resolved_path = Path(db_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        db_path = str(resolved_path)

    vector_store = MilvusVectorStore(
        uri=db_path,
        collection_name=collection_name,
        dim=embedding_dim,               # Must match your HF model dimension (e.g., BGE-small is 384)
        overwrite=overwrite,
        similarity_metric="COSINE",
    )
    return vector_store


def load_milvus_vector_store(db_path: str, collection_name: str) -> BasePydanticVectorStore:
    if not db_path.startswith("http://") and not db_path.startswith("https://"):
        db_path = str(Path(db_path).resolve())

    vector_store = MilvusVectorStore(
        uri=db_path,
        collection_name=collection_name,
        dim=None,  # Dimension is not needed for loading an existing store
        overwrite=False,
        similarity_metric="COSINE",
    )
    return vector_store


def load_neo4j_vector_store(
    url: str,
    username: str,
    password: str,
    index_name: str = "vector",
) -> BasePydanticVectorStore:
    # node_label, embedding_node_property and dimension are read back off the index once it matches by name.
    vector_store = Neo4jVectorStore(
        username=username,
        password=password,
        url=url,
        embedding_dimension=_INDEX_INFERRED_DIM,
        index_name=index_name,
        distance_strategy="cosine",
        retrieval_query=_NEO4J_RETRIEVAL_QUERY,
    )
    return vector_store


def create_index_from_embedded_chunks(
    vector_store: BasePydanticVectorStore,
    embedded_chunks: list[BaseNode],
) -> VectorStoreIndex:
    assert len(embedded_chunks) > 0, "No embedded chunks provided to create the index."
    assert all(hasattr(chunk, "embedding") and chunk.embedding is not None for chunk in embedded_chunks), "All chunks must have embeddings before creating the index."

    # Stores expose their dimension under different names (Milvus: dim, Neo4j: embedding_dimension).
    store_dim = getattr(vector_store, "dim", None) or getattr(vector_store, "embedding_dimension", None)
    embedding = embedded_chunks[0].embedding
    if store_dim is not None:
        assert embedding is not None and len(embedding) == store_dim, f"Embedding dimension of chunks ({len(embedding) if embedding else 0}) does not match vector store dimension ({store_dim})."

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=embedded_chunks,
        storage_context=storage_context,
        show_progress=False,
        embed_model=None,
    )

    return index
