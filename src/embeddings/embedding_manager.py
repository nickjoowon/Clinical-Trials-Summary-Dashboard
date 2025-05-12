from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding manager with a specific model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True)
    
    def build_index(self, embeddings: np.ndarray):
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def add_metadata(self, metadata: List[Dict[str, Any]]):
        """Add metadata for each embedding."""
        self.metadata = metadata
    
    def save_index(self, directory: str):
        """Save the FAISS index and metadata to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save the metadata
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
    
    def load_index(self, directory: str):
        """Load the FAISS index and metadata from disk."""
        # Load the index
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load the metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts using a query string."""
        # Create embedding for the query
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Ensure index is valid
                result = self.metadata[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results 