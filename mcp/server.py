import argparse

from mcp.server.mcpserver import MCPServer

from pipeline.run_pipeline import run_pipeline

server = MCPServer("RAG Retrieval Server")


@server.tool("retrieve_chunks")
def retrieve_chunks(query: str) -> list[str]:
    """Tool to retrieve relevant chunks for a given query."""
    return run_pipeline(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the RAG Pipeline MCP Server.")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to run the MCP server on (default: 8001)"
    )
    args = parser.parse_args()
    server.run(transport="streamable-http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
