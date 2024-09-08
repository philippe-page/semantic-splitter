import os
from typing import List
import numpy as np
import igraph as ig
import leidenalg as la
import cohere
from dotenv import load_dotenv
from sentence_splitter import SentenceSplitter

load_dotenv()

class SemanticSplitter:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY must be set in environment variables.")
        self.co = cohere.Client(self.api_key)
        self.embedding_model = 'embed-english-v3.0'
        self.splitter = SentenceSplitter(language='en')

    @staticmethod
    def chunk_text(text: str, resolution: float = 1.0) -> List[str]:
        splitter = SemanticSplitter()
        segments = splitter._create_sentence_segments(text)
        embeddings = splitter._embed_segments(segments)
        breakpoints = splitter._detect_communities(embeddings, resolution)
        chunks = splitter._create_chunks_from_communities(segments, breakpoints)
        
        print(f"Created {len(chunks)} non-empty chunks")
        return chunks

    def _create_sentence_segments(self, text: str) -> List[str]:
        sentences = self.splitter.split(text)
        segments = [sentence.strip() for sentence in sentences]
        print(f"Created {len(segments)} segments")
        return segments

    def _embed_segments(self, segments: List[str]) -> np.ndarray:
        batch_size = 96  # Cohere's maximum batch size
        embeddings = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            response = self.co.embed(texts=batch, model=self.embedding_model, input_type='search_document')
            embeddings.extend(response.embeddings)
        return np.array(embeddings)

    def _detect_communities(self, embeddings: np.ndarray, resolution: float) -> List[int]:
        if embeddings.shape[0] < 2:
            return [0]
        
        G = self._create_similarity_graph(embeddings)
        
        partition = self._find_optimal_partition(G, resolution)
        
        communities = partition.membership
        
        # Identify breakpoints where community changes
        breakpoints = self._identify_breakpoints(communities)
        
        num_communities = len(breakpoints) + 1
        print(f"Resolution: {resolution}, Communities: {num_communities}")
        
        return breakpoints

    def _identify_breakpoints(self, communities: List[int]) -> List[int]:
        breakpoints = []
        for i in range(1, len(communities)):
            if communities[i] != communities[i-1]:
                breakpoints.append(i)
        return breakpoints

    def _create_similarity_graph(self, embeddings: np.ndarray) -> ig.Graph:
        G = ig.Graph.Full(embeddings.shape[0])
        similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarities, 0)
        similarities = np.maximum(similarities, 0)
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
        G.es['weight'] = similarities[np.triu_indices(embeddings.shape[0], k=1)]
        return G

    def _find_optimal_partition(self, G: ig.Graph, resolution: float) -> la.VertexPartition:
        return la.find_partition(
            G, 
            la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution
        )

    def _split_oversized_communities(self, membership: List[int], max_size: int) -> List[int]:
        community_sizes = {}
        for comm in membership:
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        
        new_membership = []
        current_comm = max(membership) + 1
        for i, comm in enumerate(membership):
            if community_sizes[comm] > max_size:
                if i % max_size == 0:
                    current_comm += 1
                new_membership.append(current_comm)
            else:
                new_membership.append(comm)
        
        return new_membership

    def _create_chunks_from_communities(self, segments: List[str], breakpoints: List[int]) -> List[str]:
        chunks = []
        start = 0
        for end in breakpoints:
            chunk = ' '.join(segments[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        # Add the last chunk
        last_chunk = ' '.join(segments[start:]).strip()
        if last_chunk:
            chunks.append(last_chunk)
        
        return chunks
