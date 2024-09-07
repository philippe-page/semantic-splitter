import os
import numpy as np
import cohere
import networkx as nx
from community import community_louvain
from typing import List, Optional
from sentence_splitter import SentenceSplitter
import logging

class SemanticSplitter:
    def __init__(self, api_key: Optional[str] = None, similarity_threshold: float = 0.2):
        if api_key is None:
            api_key = os.environ.get('COHERE_API_KEY')
        if api_key is None:
            raise ValueError("Cohere API key must be provided or set as COHERE_API_KEY environment variable.")
        self.co = cohere.Client(api_key)

        self.embedding_model = 'embed-english-v3.0'
        self.similarity_threshold = similarity_threshold

        # Initialize the sentence splitter
        self.splitter = SentenceSplitter(language='en')

    def chunk_text(self, text: str) -> List[str]:
        # Step 1: Create segments (now using sentences instead of fixed-length segments)
        segments = self._create_sentence_segments(text)
        
        # Step 2: Embed each segment
        embeddings = self._embed_segments(segments)
        print(f"Shape of embeddings: {embeddings.shape}")
        
        # Step 3: Detect major topic boundaries
        boundaries = self._detect_major_topic_boundaries(embeddings)
        
        # Step 4: Create chunks based on detected boundaries
        chunks = self._create_chunks_from_boundaries(segments, boundaries)
        
        print(f"Number of final chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} length: {len(chunk)} characters")
        
        return chunks, embeddings

    def _create_sentence_segments(self, text: str) -> List[str]:
        # Use sentence-splitter to split the text into sentences
        sentences = self.splitter.split(text)
        
        # Remove the combining of short segments
        segments = [sentence.strip() for sentence in sentences]
        
        print(f"Created {len(segments)} segments")
        for i, segment in enumerate(segments):
            print(f"Segment {i+1} length: {len(segment)} characters")
        
        return segments

    def _embed_segments(self, segments: List[str]) -> np.ndarray:
        embeddings = []
        for segment in segments:
            response = self.co.embed(texts=[segment], model=self.embedding_model, input_type='search_document')
            embeddings.append(response.embeddings[0])
        return np.array(embeddings)

    def _detect_major_topic_boundaries(self, embeddings: np.ndarray) -> List[int]:
        if embeddings.shape[0] < 2:
            return []
        
        # Create a fully connected graph
        G = nx.Graph()
        for i in range(embeddings.shape[0]):
            for j in range(i+1, embeddings.shape[0]):
                similarity = np.dot(embeddings[i], embeddings[j])
                if similarity > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # Ensure all nodes are in the graph
        G.add_nodes_from(range(embeddings.shape[0]))
        
        # Apply Louvain community detection
        communities = community_louvain.best_partition(G)
        
        # Find community boundaries
        boundaries = []
        prev_community = communities.get(0, 0)  # Default to 0 if not found
        for i in range(1, embeddings.shape[0]):
            current_community = communities.get(i, prev_community)  # Use previous if not found
            if current_community != prev_community:
                boundaries.append(i)
            prev_community = current_community
        
        print(f"Detected boundaries: {boundaries}")
        return boundaries

    def _create_chunks_from_boundaries(self, segments: List[str], boundaries: List[int]) -> List[str]:
        """Create chunks based on detected boundaries and remove empty chunks."""
        chunks = []
        start = 0
        for boundary in boundaries + [len(segments)]:
            chunk = ' '.join(segments[start:boundary]).strip()
            if chunk:
                chunks.append(chunk)
                logging.info(f"Created chunk of length {len(chunk)} characters")
            else:
                logging.warning(f"Skipped empty chunk between segments {start} and {boundary}")
            start = boundary
        
        logging.info(f"Created {len(chunks)} non-empty chunks")
        return chunks

def chunk_text(text: str, api_key: Optional[str] = None, similarity_threshold: float = 0.2) -> List[str]:
    """
    A convenience function to chunk text without explicitly creating a SemanticSplitter instance.

    Args:
        text (str): The input text to be chunked.
        api_key (Optional[str]): Cohere API key. If None, it will try to use the COHERE_API_KEY environment variable.
        similarity_threshold (float): The similarity threshold for detecting topic boundaries. Default is 0.6.

    Returns:
        List[str]: A list of text chunks.
    """
    chunker = SemanticSplitter(api_key=api_key, similarity_threshold=similarity_threshold)
    chunks, _ = chunker.chunk_text(text)
    return chunks

