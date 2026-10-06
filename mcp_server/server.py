import argparse
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from pipeline.run_pipeline import run_pipeline

server = MCPServer("RAG Retrieval Server")

LOG_PATH = Path(os.environ.get("ENGRAM_RAG_LOG", "logs/input-output.jsonl"))
_log_lock = threading.Lock()


def _log_exchange(query: str, chunks: list[str] | None, error: str | None) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "chunk_count": len(chunks) if chunks is not None else 0,
        "chunks": chunks or [],
        "error": error,
    }
    with _log_lock:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")


@server.tool("retrieve_chunks")
def retrieve_chunks(query: str) -> list[str]:
    """Tool to retrieve relevant chunks for a given query."""
    try:
        chunks = run_pipeline(query)
    except Exception as error:
        _log_exchange(query, None, f"{type(error).__name__}: {error}")
        raise
    _log_exchange(query, chunks, None)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the RAG Pipeline MCP Server.")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to run the MCP server on (default: 8001)"
    )
    args = parser.parse_args()
    server.run(transport="streamable-http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
