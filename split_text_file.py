from semantic_splitter import SemanticSplitter

def split_and_print_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Create an instance of SemanticSplitter
    splitter = SemanticSplitter()

    # Call chunk_text on the instance
    chunks = splitter.chunk_text(text, resolution=0.34)

    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

if __name__ == "__main__":

    file_path = "test_text.txt"
    split_and_print_chunks(file_path)
    print("\n\n\n===========\n\n\n")


    