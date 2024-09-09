# SemanticSplitter

SemanticSplitter is a Python tool that chunks text based on semantic similarity. It uses sentence-level segmentation and the Cohere API for embedding generation, combined with community detection algorithms to identify topic boundaries.

## Features

- Sentence-level text segmentation
- Semantic similarity-based chunking
- Adjustable resolution parameter for community detection
- Adjustable similarity threshold for graph construction
- Option to rearrange chunks based on semantic similarity
- Integration with Cohere's embedding API
- Community detection using the Leiden algorithm

## How It Works

The SemanticSplitter class implements a sophisticated approach to chunking unstructured text based on semantic similarity and topic boundaries. This method is particularly useful for processing long documents or articles where traditional fixed-length chunking might break apart coherent ideas or combine unrelated concepts.

The process works as follows:

1. The text is first split into sentences using a SentenceSplitter, which provides a more natural segmentation than arbitrary character-based splits.

2. Each sentence is then embedded using Cohere's embedding model. These embeddings capture the semantic meaning of each sentence in a high-dimensional vector space.

3. A similarity graph is constructed where each node represents a sentence, and edges are weighted based on the cosine similarity between sentence embeddings. A similarity threshold is applied to determine which edges are included in the graph.

4. The Leiden community detection algorithm is applied to this graph. This algorithm is particularly good at finding communities (clusters) in networks, which in this context correspond to coherent topics or themes in the text.

5. By identifying where these communities change, the algorithm detects major topic boundaries in the text. These boundaries are then used to create chunks of text that are semantically coherent.

6. Optionally, the chunks can be rearranged based on their semantic similarity, grouping similar topics together regardless of their original order in the text.

## Benefits

The benefits of this approach are numerous:

1. **Semantic Coherence**: Unlike fixed-length chunking, this method ensures that each chunk contains semantically related content.

2. **Adaptability**: The algorithm adapts to the natural structure of the text, creating larger chunks for areas with consistent topics and smaller chunks where topics change rapidly.

3. **Customizability**: The resolution parameter and similarity threshold allow fine-tuning of the chunking process to suit different types of text and use cases.

4. **Language Model Friendly**: By creating semantically coherent chunks, this method is particularly well-suited for use with large language models, which can better understand and process coherent pieces of text.

5. **Improved Information Retrieval**: When used in conjunction with search or retrieval systems, these semantic chunks can lead to more relevant and contextually appropriate results.

6. **Scalability**: The use of sentence embeddings and graph-based community detection allows this method to scale to very large documents while still capturing fine-grained semantic structure.

## Prerequisites

- Python 3.7+
- Cohere API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/philippe-page/semantic-splitter.git
   cd semantic-splitter
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Cohere API key:
   - Create a `.env` file in the project root directory:
     ```
     COHERE_API_KEY=your-api-key-here
     ```
   - Make sure to add `.env` to your `.gitignore` file to keep your API key secure

## Usage

```python
from semantic_splitter import SemanticSplitter

# Initialize SemanticSplitter
splitter = SemanticSplitter()

# Chunk the text with optional parameters
text = "Your long text here..."
chunks = SemanticSplitter.chunk_text(
    text,
    resolution=1.0,
    similarity_threshold=0.5,
    rearrange=False
)

# Process the chunks
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(chunk)
    print("---")
```

Parameters:
- `resolution`: Controls the granularity of the community detection. Higher values result in more communities (and thus more chunks), while lower values produce fewer, larger communities.
- `similarity_threshold`: Determines the minimum similarity required for two sentences to be connected in the graph. Higher values create a sparser graph with more distinct communities.
- `rearrange`: If set to True, chunks will be rearranged based on their semantic similarity, grouping similar topics together regardless of their original order in the text.

## Contributing

This is still a work in progress, so feel free to submit a PR.

## License

This project is licensed under the MIT License.