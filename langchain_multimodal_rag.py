"""
LangChain-Compatible Multimodal RAG Implementation
A comprehensive multimodal RAG system built with LangChain components.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
import requests
from io import BytesIO
import base64

# LangChain imports
from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.retrievers.base import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever as LangChainBaseRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema.runnable import Runnable

# Transformers for multimodal processing
try:
    from transformers import (
        CLIPProcessor, CLIPModel,
        BlipProcessor, BlipForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Using simplified embeddings.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalDocument:
    """Enhanced document class compatible with LangChain."""
    id: str
    text: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_langchain_document(self) -> LangChainDocument:
        """Convert to LangChain Document format."""
        metadata = self.metadata or {}
        metadata.update({
            "id": self.id,
            "has_image": bool(self.image_path or self.image_url),
            "image_path": self.image_path,
            "image_url": self.image_url
        })
        return LangChainDocument(page_content=self.text, metadata=metadata)

class MultimodalEmbeddings(Embeddings):
    """LangChain-compatible multimodal embeddings class."""
    
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        caption_model: str = "Salesforce/blip-image-captioning-base"
    ):
        """Initialize multimodal embeddings."""
        self.text_model_name = text_model
        self.vision_model_name = vision_model
        self.caption_model_name = caption_model
        
        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            logger.warning("Using simplified embeddings due to missing dependencies")
    
    def _load_models(self):
        """Load the required models."""
        try:
            # Text embedding model
            self.text_encoder = SentenceTransformer(self.text_model_name)
            
            # Vision-language model (CLIP)
            self.clip_processor = CLIPProcessor.from_pretrained(self.vision_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.vision_model_name)
            
            # Image captioning model (BLIP)
            self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(self.caption_model_name)
            
            logger.info("All multimodal models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            TRANSFORMERS_AVAILABLE = False
    
    def _simple_text_embedding(self, text: str) -> List[float]:
        """Simple text embedding fallback."""
        words = text.lower().split()
        vocab = set(words)
        
        features = []
        features.append(len(words))
        features.append(len(vocab))
        features.append(len(text))
        
        # Character frequency features
        char_counts = {}
        for char in text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(char_counts.get(char, 0))
        
        # Pad to consistent length
        while len(features) < 384:
            features.append(0.0)
        
        return features[:384]
    
    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, Image.Image):
            return image_source
        elif isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                return Image.open(image_source).convert('RGB')
        else:
            raise ValueError("Image source must be a file path, URL, or PIL Image")
    
    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate caption for an image."""
        if not TRANSFORMERS_AVAILABLE:
            # Simple fallback description
            width, height = image.size
            return f"Image ({width}x{height} pixels)"
        
        try:
            inputs = self.caption_processor(image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"Image processing failed: {str(e)}"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            if TRANSFORMERS_AVAILABLE:
                try:
                    embedding = self.text_encoder.encode(text)
                    embeddings.append(embedding.tolist())
                except Exception as e:
                    logger.error(f"Error embedding text: {e}")
                    embeddings.append(self._simple_text_embedding(text))
            else:
                embeddings.append(self._simple_text_embedding(text))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if TRANSFORMERS_AVAILABLE:
            try:
                embedding = self.text_encoder.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error embedding query: {e}")
                return self._simple_text_embedding(text)
        else:
            return self._simple_text_embedding(text)
    
    def embed_multimodal_document(self, document: MultimodalDocument) -> List[float]:
        """Embed a multimodal document with text and optional image."""
        text_content = document.text
        
        # Process image if available
        if document.image_path or document.image_url:
            try:
                image_source = document.image_path or document.image_url
                image = self.load_image(image_source)
                image_caption = self.generate_image_caption(image)
                text_content = f"{document.text} [Image: {image_caption}]"
            except Exception as e:
                logger.warning(f"Could not process image for document {document.id}: {e}")
        
        return self.embed_query(text_content)

class MultimodalRetriever(BaseRetriever):
    """LangChain-compatible multimodal retriever."""
    
    def __init__(
        self,
        vectorstore: Chroma,
        embeddings: MultimodalEmbeddings,
        k: int = 5,
        search_type: str = "similarity"
    ):
        """Initialize the multimodal retriever."""
        super().__init__()
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.k = k
        self.search_type = search_type
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[LangChainDocument]:
        """Retrieve relevant documents for a query."""
        try:
            if self.search_type == "similarity":
                docs = self.vectorstore.similarity_search(query, k=self.k)
            elif self.search_type == "mmr":
                docs = self.vectorstore.max_marginal_relevance_search(query, k=self.k)
            else:
                docs = self.vectorstore.similarity_search(query, k=self.k)
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

class SimpleLLM(LLM):
    """Simple LLM implementation for demonstration purposes."""
    
    def __init__(self, model_name: str = "simple"):
        super().__init__()
        self.model_name = model_name
    
    @property
    def _llm_type(self) -> str:
        return "simple"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Simple response generation."""
        # This is a placeholder - in practice, you'd use a real LLM
        return f"Based on the provided context, here's a response to your query. This is a simple demonstration response. In a real implementation, you would integrate with models like OpenAI GPT, Anthropic Claude, or local models via Ollama."

class LangChainMultimodalRAG:
    """Main LangChain-compatible multimodal RAG system."""
    
    def __init__(
        self,
        collection_name: str = "langchain_multimodal_rag",
        persist_directory: str = "./langchain_chroma_db",
        text_splitter_chunk_size: int = 1000,
        text_splitter_chunk_overlap: int = 200
    ):
        """Initialize the LangChain multimodal RAG system."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embeddings = MultimodalEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_chunk_size,
            chunk_overlap=text_splitter_chunk_overlap
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Initialize retriever
        self.retriever = MultimodalRetriever(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings
        )
        
        # Initialize LLM (placeholder - replace with real LLM)
        self.llm = SimpleLLM()
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        logger.info("LangChain Multimodal RAG system initialized successfully")
    
    def add_documents(self, documents: List[MultimodalDocument]) -> bool:
        """Add multiple documents to the system."""
        try:
            langchain_docs = []
            
            for doc in documents:
                # Process multimodal content
                text_content = doc.text
                
                # Add image information if available
                if doc.image_path or doc.image_url:
                    try:
                        image_source = doc.image_path or doc.image_url
                        image = self.embeddings.load_image(image_source)
                        image_caption = self.embeddings.generate_image_caption(image)
                        text_content = f"{doc.text}\n\nImage Description: {image_caption}"
                    except Exception as e:
                        logger.warning(f"Could not process image for document {doc.id}: {e}")
                
                # Split text if needed
                text_chunks = self.text_splitter.split_text(text_content)
                
                # Create LangChain documents
                for i, chunk in enumerate(text_chunks):
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    metadata.update({
                        "id": f"{doc.id}_chunk_{i}",
                        "original_id": doc.id,
                        "chunk_index": i,
                        "has_image": bool(doc.image_path or doc.image_url),
                        "image_path": doc.image_path,
                        "image_url": doc.image_url
                    })
                    
                    langchain_doc = LangChainDocument(
                        page_content=chunk,
                        metadata=metadata
                    )
                    langchain_docs.append(langchain_doc)
            
            # Add to vector store
            self.vectorstore.add_documents(langchain_docs)
            
            # Persist the vector store
            self.vectorstore.persist()
            
            logger.info(f"Successfully added {len(documents)} documents ({len(langchain_docs)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def add_document(self, document: MultimodalDocument) -> bool:
        """Add a single document to the system."""
        return self.add_documents([document])
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant documents."""
        try:
            docs = self.retriever.get_relevant_documents(query)
            
            results = []
            for doc in docs[:k]:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)
                })
            
            return {
                "query": query,
                "results": results,
                "num_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"query": query, "results": [], "num_results": 0}
    
    def query(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embeddings.text_model_name,
                "vision_model": self.embeddings.vision_model_name,
                "transformers_available": TRANSFORMERS_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete and recreate the collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Update retriever
            self.retriever.vectorstore = self.vectorstore
            
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

# Convenience functions for easy integration
def create_multimodal_rag(
    collection_name: str = "multimodal_rag",
    persist_directory: str = "./chroma_db"
) -> LangChainMultimodalRAG:
    """Create a new multimodal RAG system."""
    return LangChainMultimodalRAG(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

def create_document(
    id: str,
    text: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    **metadata
) -> MultimodalDocument:
    """Create a new multimodal document."""
    return MultimodalDocument(
        id=id,
        text=text,
        image_path=image_path,
        image_url=image_url,
        metadata=metadata
    )

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing LangChain Multimodal RAG System")
    
    # Create RAG system
    rag = create_multimodal_rag("demo_collection")
    
    # Create sample documents
    docs = [
        create_document(
            id="ai_doc_1",
            text="Artificial intelligence and machine learning are revolutionizing technology. Deep learning models can process vast amounts of data to identify patterns and make predictions.",
            category="Technology",
            topic="AI"
        ),
        create_document(
            id="nature_doc_1",
            text="The Amazon rainforest is one of the most biodiverse ecosystems on Earth. It contains thousands of species of plants, animals, and insects, many of which are found nowhere else.",
            category="Nature",
            topic="Environment"
        ),
        create_document(
            id="space_doc_1",
            text="The James Webb Space Telescope has captured incredible images of distant galaxies, providing new insights into the formation and evolution of the universe.",
            category="Science",
            topic="Astronomy"
        )
    ]
    
    # Add documents
    print("📚 Adding documents...")
    success = rag.add_documents(docs)
    print(f"Documents added: {success}")
    
    # Test search
    print("\n🔍 Testing search functionality...")
    search_results = rag.search("machine learning and AI", k=3)
    print(f"Search found {search_results['num_results']} results")
    
    for i, result in enumerate(search_results['results']):
        print(f"  {i+1}. {result['content'][:100]}...")
        print(f"     Category: {result['metadata'].get('category', 'N/A')}")
    
    # Test Q&A
    print("\n❓ Testing Q&A functionality...")
    qa_result = rag.query("What is artificial intelligence?")
    print(f"Question: {qa_result['question']}")
    print(f"Answer: {qa_result['answer'][:200]}...")
    print(f"Sources: {len(qa_result['source_documents'])} documents")
    
    # Get stats
    print("\n📊 System statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ LangChain Multimodal RAG system is working!")
