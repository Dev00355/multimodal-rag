"""
Simplified Multimodal RAG Implementation
A working version that handles dependency conflicts gracefully.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document with text and optional image content."""
    id: str
    text: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SimpleMultimodalRAG:
    """A simplified multimodal RAG system that works with basic dependencies."""
    
    def __init__(self, collection_name: str = "simple_rag"):
        """Initialize the simple RAG system."""
        self.collection_name = collection_name
        self.documents = {}  # Simple in-memory storage
        self.embeddings = {}  # Simple embedding storage
        
        logger.info("SimpleMultimodalRAG initialized successfully")
    
    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from file path, URL, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source
        elif isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_source)
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from file path
                return Image.open(image_source).convert('RGB')
        else:
            raise ValueError("Image source must be a file path, URL, or PIL Image")
    
    def simple_text_embedding(self, text: str) -> np.ndarray:
        """Create a simple text embedding using character frequencies."""
        # Simple bag-of-words style embedding
        words = text.lower().split()
        vocab = set(words)
        
        # Create a simple feature vector
        features = []
        
        # Word count features
        features.append(len(words))
        features.append(len(vocab))
        features.append(len(text))
        
        # Character frequency features
        char_counts = {}
        for char in text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Add top character frequencies
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(char_counts.get(char, 0))
        
        # Add some simple linguistic features
        features.append(text.count('.'))  # Sentence count approximation
        features.append(text.count('?'))  # Question count
        features.append(text.count('!'))  # Exclamation count
        
        return np.array(features, dtype=np.float32)
    
    def simple_image_description(self, image: Image.Image) -> str:
        """Generate a simple image description based on basic properties."""
        width, height = image.size
        mode = image.mode
        
        # Simple description based on image properties
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            orientation = "wide landscape"
        elif aspect_ratio < 0.67:
            orientation = "tall portrait"
        else:
            orientation = "square or balanced"
        
        size_desc = "large" if width * height > 500000 else "medium" if width * height > 100000 else "small"
        
        return f"A {size_desc} {orientation} {mode.lower()} image ({width}x{height} pixels)"
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the RAG system."""
        try:
            # Prepare text content
            text_content = document.text
            
            # Process image if provided
            image_description = ""
            if document.image_path or document.image_url:
                try:
                    image_source = document.image_path or document.image_url
                    image = self.load_image(image_source)
                    image_description = self.simple_image_description(image)
                    
                    # Combine text and image description
                    text_content = f"{document.text} [Image: {image_description}]"
                except Exception as e:
                    logger.warning(f"Could not process image for document {document.id}: {e}")
            
            # Generate simple embedding
            embedding = self.simple_text_embedding(text_content)
            
            # Store document and embedding
            self.documents[document.id] = {
                "text": text_content,
                "original_text": document.text,
                "image_description": image_description,
                "metadata": document.metadata or {},
                "has_image": bool(document.image_path or document.image_url)
            }
            self.embeddings[document.id] = embedding
            
            logger.info(f"Document {document.id} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            return False
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Search for relevant documents based on a query."""
        try:
            if not self.documents:
                return {"query": query, "results": [], "num_results": 0}
            
            # Generate query embedding
            query_embedding = self.simple_text_embedding(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, embedding in self.embeddings.items():
                similarity = self.cosine_similarity(query_embedding, embedding)
                similarities.append((doc_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            top_results = similarities[:n_results]
            
            # Format results
            results = []
            for doc_id, similarity in top_results:
                doc_data = self.documents[doc_id]
                result = {
                    "id": doc_id,
                    "text": doc_data["text"],
                    "similarity": similarity,
                }
                
                if include_metadata:
                    result["metadata"] = doc_data["metadata"]
                    result["has_image"] = doc_data["has_image"]
                    result["image_description"] = doc_data["image_description"]
                
                results.append(result)
            
            return {
                "query": query,
                "results": results,
                "num_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"query": query, "results": [], "num_results": 0}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "storage_type": "in-memory"
        }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.documents.clear()
            self.embeddings.clear()
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Simple Multimodal RAG System")
    
    processor = SimpleMultimodalRAG()
    
    # Example documents
    docs = [
        Document(
            id="ai_doc",
            text="Artificial intelligence and machine learning are transforming technology.",
            metadata={"category": "AI", "author": "Demo"}
        ),
        Document(
            id="nature_doc",
            text="The rainforest contains incredible biodiversity and natural beauty.",
            metadata={"category": "Nature", "author": "Demo"}
        )
    ]
    
    # Add documents
    for doc in docs:
        success = processor.add_document(doc)
        print(f"Added {doc.id}: {success}")
    
    # Test search
    queries = ["machine learning", "forest biodiversity"]
    
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = processor.search(query, n_results=2)
        
        print(f"Found {results['num_results']} results:")
        for i, result in enumerate(results['results']):
            print(f"  {i+1}. Similarity: {result['similarity']:.3f}")
            print(f"     Text: {result['text'][:100]}...")
            print(f"     Category: {result['metadata'].get('category', 'N/A')}")
    
    # Get stats
    stats = processor.get_collection_stats()
    print(f"\n📊 Collection stats: {stats}")
    
    print("\n✅ Simple RAG system is working!")
