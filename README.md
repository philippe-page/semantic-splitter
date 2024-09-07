# SemanticSplitter

SemanticSplitter is a Python tool that chunks text based on semantic similarity. It uses sentence-level segmentation and the Cohere API for embedding generation, combined with community detection algorithms to identify topic boundaries.

## Features

- Sentence-level text segmentation
- Semantic similarity-based chunking
- Adjustable similarity threshold
- Integration with Cohere's embedding API
- Community detection for identifying topic boundaries


## How It Works

The SemanticSplitter class implements a sophisticated approach to chunking unstructured text based on semantic similarity and topic boundaries. This method is particularly useful for processing long documents or articles where traditional fixed-length chunking might break apart coherent ideas or combine unrelated concepts.

The process works as follows:

1. The text is first split into sentences using a SentenceSplitter, which provides a more natural segmentation than arbitrary character-based splits.

2. Each sentence is then embedded using Cohere's embedding model. These embeddings capture the semantic meaning of each sentence in a high-dimensional vector space.

3. The core of the algorithm lies in the `_detect_major_topic_boundaries` method. It constructs a graph where each node represents a sentence, and edges are drawn between sentences that are semantically similar (based on the dot product of their embeddings exceeding a threshold). This graph representation allows for a more nuanced understanding of the text's structure.

4. The Leiden community detection algorithm is applied to this graph. This algorithm is particularly good at finding communities (clusters) in networks, which in this context correspond to coherent topics or themes in the text.

5. By identifying where these communities change, the algorithm detects major topic boundaries in the text. These boundaries are then used to create chunks of text that are semantically coherent.

## Benefits

The benefits of this approach are numerous:

1. **Semantic Coherence**: Unlike fixed-length chunking, this method ensures that each chunk contains semantically related content, making it more useful for downstream tasks like summarization or question-answering.

2. **Adaptability**: The algorithm adapts to the natural structure of the text, creating larger chunks for areas with consistent topics and smaller chunks where topics change rapidly.

3. **Language Model Friendly**: By creating semantically coherent chunks, this method is particularly well-suited for use with large language models, which can better understand and process coherent pieces of text.

4. **Improved Information Retrieval**: When used in conjunction with search or retrieval systems, these semantic chunks can lead to more relevant and contextually appropriate results.

5. **Scalability**: The use of sentence embeddings and graph-based community detection allows this method to scale to very large documents while still capturing fine-grained semantic structure.


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

4. Load the environment variables:
   - Install the `python-dotenv` package:
     ```
     pip install python-dotenv
     ```
   - In your Python script, load the environment variables:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```

## Usage
    ```python
    from semantic_splitter import SemanticSplitter

    # Initialize SemanticSplitter with the specified chunk size
    splitter = SemanticSplitter(chunk_size=chunk_size)

    # Chunk the text
    chunks, communities = splitter.chunk_text(text)
    ```

## Contributing

This is still a work in progress so feel free to submit a PR.

## License

This project is licensed under the MIT License.
