import os
import numpy as np
import cohere
from typing import List, Optional, Tuple
from sentence_splitter import SentenceSplitter
import logging
import leidenalg
import igraph as ig

class SemanticSplitter:
    def __init__(self, api_key: Optional[str] = None, chunk_size: str = 'medium'):
        if api_key is None:
            api_key = os.environ.get('COHERE_API_KEY')
        if api_key is None:
            raise ValueError("Cohere API key must be provided or set as COHERE_API_KEY environment variable.")
        self.co = cohere.Client(api_key)

        self.embedding_model = 'embed-english-v3.0'
        self.chunk_params = {
            'fine': {'resolution_parameter': 2, 'min_chunk_size': 200},
            'medium': {'resolution_parameter': 1.5, 'min_chunk_size': 300},
            'large': {'resolution_parameter': 1, 'min_chunk_size': 400}
        }
        self.set_chunk_size(chunk_size)

        # Initialize the sentence splitter
        self.splitter = SentenceSplitter(language='en')

    def set_chunk_size(self, chunk_size: str):
        params = self.chunk_params.get(chunk_size, self.chunk_params['medium'])
        self.resolution_parameter = params['resolution_parameter']
        self.min_chunk_size = params['min_chunk_size']

    def chunk_text(self, text: str) -> Tuple[List[str], List[int]]:
        # Step 1: Create segments (now using sentences instead of fixed-length segments)
        segments = self._create_sentence_segments(text)
        
        # Step 2: Embed each segment
        embeddings = self._embed_segments(segments)
        print(f"Shape of embeddings: {embeddings.shape}")
        
        # Step 3: Detect major topic boundaries
        boundaries = self._detect_major_topic_boundaries(embeddings)
        
        # Step 4: Create chunks based on detected boundaries
        chunks = self._create_chunks_from_boundaries(segments, boundaries)
        
        # Step 5: Merge small chunks
        chunks = self._merge_small_chunks(chunks, segments)
        
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
        G = ig.Graph.Full(embeddings.shape[0])
        weights = []
        for i in range(embeddings.shape[0]):
            for j in range(i+1, embeddings.shape[0]):
                similarity = np.dot(embeddings[i], embeddings[j])
                weights.append(similarity)
        
        G.es['weight'] = weights
        
        # Apply Leiden community detection with resolution parameter
        partition = leidenalg.find_partition(
            G, 
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=self.resolution_parameter
        )
        
        # Find community boundaries
        boundaries = []
        prev_community = partition.membership[0]
        for i in range(1, embeddings.shape[0]):
            current_community = partition.membership[i]
            if current_community != prev_community:
                boundaries.append(i)
            prev_community = current_community
        
        print(f"Detected boundaries: {boundaries}")
        print(f"Total communities: {len(set(partition.membership))}")
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

    def _merge_small_chunks(self, chunks: List[str], segments: List[str]) -> List[str]:
        merged_chunks = []
        current_chunk = ""
        segment_index = 0

        for chunk in chunks:
            if len(current_chunk) + len(chunk) < self.min_chunk_size:
                current_chunk += " " + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    # If current_chunk is still too small, add more sentences
                    while len(current_chunk) < self.min_chunk_size and segment_index < len(segments):
                        current_chunk += " " + segments[segment_index]
                        segment_index += 1
                    merged_chunks.append(current_chunk.strip())
                current_chunk = chunk

        # Handle the last chunk
        if current_chunk:
            while len(current_chunk) < self.min_chunk_size and segment_index < len(segments):
                current_chunk += " " + segments[segment_index]
                segment_index += 1
            merged_chunks.append(current_chunk.strip())

        return merged_chunks

def chunk_text(text: str, api_key: Optional[str] = None, chunk_size: str = 'medium') -> List[str]:
    chunker = SemanticSplitter(api_key=api_key, chunk_size=chunk_size)
    chunks, _ = chunker.chunk_text(text)
    return chunks

