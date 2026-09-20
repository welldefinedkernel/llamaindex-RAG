from pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    try:
        while True:
            input_query = input("Enter your query: ")
            chunks = run_pipeline(input_query)
            if not chunks:
                print("No matching chunks.\n")
                continue
            for i, chunk in enumerate(chunks, start=1):
                print(f"\n{'=' * 70}")
                print(f"[{i}/{len(chunks)}]")
                print("-" * 70)
                print(chunk.strip())
            print()
    except (EOFError, KeyboardInterrupt):
        print()
