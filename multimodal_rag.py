"""
Multimodal RAG Implementation
A comprehensive multimodal RAG system that can process text and images.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModel
)
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
from io import BytesIO
import base64

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

class MultimodalRAGProcessor:
    """A comprehensive multimodal RAG system that can process text and images."""
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        caption_model: str = "Salesforce/blip-image-captioning-base",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the MultimodalRAGProcessor.
        
        Args:
            collection_name: Name for the ChromaDB collection
            embedding_model: Sentence transformer model for text embeddings
            vision_model: CLIP model for image-text similarity
            caption_model: BLIP model for image captioning
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize models
        logger.info("Loading models...")
        self._load_models(embedding_model, vision_model, caption_model)
        
        # Initialize ChromaDB
        self._initialize_database()
        
        logger.info("MultimodalRAGProcessor initialized successfully")
    
    def _load_models(self, embedding_model: str, vision_model: str, caption_model: str):
        """Load all required models."""
        try:
            # Text embedding model
            self.text_encoder = SentenceTransformer(embedding_model)
            
            # Vision-language model (CLIP)
            self.clip_processor = CLIPProcessor.from_pretrained(vision_model)
            self.clip_model = CLIPModel.from_pretrained(vision_model)
            
            # Image captioning model (BLIP)
            self.caption_processor = BlipProcessor.from_pretrained(caption_model)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Database initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
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
    
    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate a caption for an image using BLIP."""
        try:
            inputs = self.caption_processor(image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Caption generation failed"
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.numpy().flatten()
        except Exception as e:
            logger.error(f"Error getting image embedding: {e}")
            return np.zeros(512)  # Return zero vector as fallback
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get sentence transformer embedding for text."""
        try:
            return self.text_encoder.encode(text)
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return np.zeros(384)  # Return zero vector as fallback
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the RAG system."""
        try:
            # Prepare text content
            text_content = document.text
            
            # Process image if provided
            image_caption = ""
            if document.image_path or document.image_url:
                image_source = document.image_path or document.image_url
                image = self.load_image(image_source)
                image_caption = self.generate_image_caption(image)
                
                # Combine text and image caption
                text_content = f"{document.text} [Image: {image_caption}]"
            
            # Generate embedding
            embedding = self.get_text_embedding(text_content)
            
            # Prepare metadata
            metadata = document.metadata or {}
            metadata.update({
                "has_image": bool(document.image_path or document.image_url),
                "image_caption": image_caption,
                "original_text": document.text
            })
            
            # Add to collection
            self.collection.add(
                documents=[text_content],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[document.id]
            )
            
            logger.info(f"Document {document.id} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            return False
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Search for relevant documents based on a query."""
        try:
            # Generate query embedding
            query_embedding = self.get_text_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"] if include_metadata else ["documents"]
            )
            
            return {
                "query": query,
                "results": results,
                "num_results": len(results["documents"][0]) if results["documents"] else 0
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"query": query, "results": None, "num_results": 0}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    processor = MultimodalRAGProcessor()
    
    # Example document
    doc = Document(
        id="example_1",
        text="This is a sample document about artificial intelligence and machine learning.",
        metadata={"category": "AI", "author": "Example Author"}
    )
    
    # Add document
    processor.add_document(doc)
    
    # Search
    results = processor.search("machine learning")
    print(f"Found {results['num_results']} results for query: {results['query']}")
    
    # Get stats
    stats = processor.get_collection_stats()
    print(f"Collection stats: {stats}")
