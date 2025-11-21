"""
Vector Store - FAISS-based semantic memory search
Enables consciousness platform to remember and recall relevant context

Built by John + Claude (Anthropic)
MIT Licensed
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Optional
import threading


class VectorStore:
    """
    Semantic memory storage using FAISS for fast similarity search.
    Stores embeddings of conversations and enables context retrieval.
    """

    def __init__(self, dimension: int = 384, storage_path: Optional[Path] = None):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension (default 384 for all-MiniLM-L6-v2)
            storage_path: Path to persist index (default: ~/.llama_selfmod_memory/)
        """
        self.dimension = dimension
        self.storage_path = storage_path or (Path.home() / ".llama_selfmod_memory")
        self.storage_path.mkdir(exist_ok=True)

        # FAISS index for vector similarity search
        self.index = faiss.IndexFlatL2(dimension)

        # Metadata storage (maps index position to actual data)
        self.metadata: List[Dict] = []

        # Sentence transformer for embeddings
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Thread lock for safe concurrent access
        self.lock = threading.Lock()

        # Try to load existing index
        self.load()

    def add_memory(self, text: str, metadata: Dict) -> int:
        """
        Add a memory to the vector store.

        Args:
            text: The text to embed and store
            metadata: Associated metadata (role, timestamp, tags, etc.)

        Returns:
            Index position of the stored memory
        """
        with self.lock:
            # Generate embedding
            embedding = self.model.encode([text], convert_to_numpy=True)

            # Add to FAISS index
            self.index.add(embedding.astype('float32'))

            # Store metadata
            memory_entry = {
                'text': text,
                **metadata
            }
            self.metadata.append(memory_entry)

            # Return index position
            return len(self.metadata) - 1

    def add_memories_batch(self, texts: List[str], metadatas: List[Dict]) -> List[int]:
        """
        Add multiple memories at once (more efficient).

        Args:
            texts: List of texts to embed
            metadatas: List of metadata dicts

        Returns:
            List of index positions
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")

        with self.lock:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))

            # Store metadata
            start_idx = len(self.metadata)
            for text, metadata in zip(texts, metadatas):
                memory_entry = {
                    'text': text,
                    **metadata
                }
                self.metadata.append(memory_entry)

            # Return index positions
            return list(range(start_idx, len(self.metadata)))

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar memories.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (metadata_dict, distance_score) tuples, sorted by relevance
        """
        with self.lock:
            if self.index.ntotal == 0:
                return []

            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)

            # Search FAISS index
            k = min(k, self.index.ntotal)  # Don't request more than we have
            distances, indices = self.index.search(query_embedding.astype('float32'), k)

            # Gather results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    results.append((self.metadata[idx], float(dist)))

            return results

    def search_with_filter(self, query: str, filter_fn, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search with custom filter function.

        Args:
            query: Query text
            filter_fn: Function that takes metadata dict and returns bool
            k: Number of results to return

        Returns:
            Filtered results
        """
        # Get more results than needed, then filter
        initial_results = self.search(query, k=k * 3)

        # Apply filter
        filtered = [(meta, dist) for meta, dist in initial_results if filter_fn(meta)]

        # Return top k
        return filtered[:k]

    def get_recent_memories(self, n: int = 10) -> List[Dict]:
        """
        Get the N most recent memories.

        Args:
            n: Number of recent memories to retrieve

        Returns:
            List of metadata dicts
        """
        with self.lock:
            return self.metadata[-n:] if self.metadata else []

    def count(self) -> int:
        """Get total number of memories stored."""
        with self.lock:
            return len(self.metadata)

    def save(self):
        """Persist index and metadata to disk."""
        with self.lock:
            try:
                # Save FAISS index
                index_path = self.storage_path / "faiss.index"
                faiss.write_index(self.index, str(index_path))

                # Save metadata
                metadata_path = self.storage_path / "metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)

                print(f"✓ Memory saved: {self.count()} memories")

            except Exception as e:
                print(f"✗ Error saving memory: {e}")

    def load(self):
        """Load index and metadata from disk."""
        try:
            index_path = self.storage_path / "faiss.index"
            metadata_path = self.storage_path / "metadata.pkl"

            if index_path.exists() and metadata_path.exists():
                with self.lock:
                    # Load FAISS index
                    self.index = faiss.read_index(str(index_path))

                    # Load metadata
                    with open(metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)

                    print(f"✓ Memory loaded: {self.count()} memories")

        except Exception as e:
            print(f"✗ Error loading memory: {e}")
            # Reset to empty state
            with self.lock:
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = []

    def clear(self):
        """Clear all memories (use with caution!)."""
        with self.lock:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("✓ Memory cleared")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text (useful for debugging/analysis).

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding
        """
        return self.model.encode([text], convert_to_numpy=True)[0]
