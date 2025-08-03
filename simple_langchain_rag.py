"""
Simple LangChain-Compatible Multimodal RAG
A lightweight version that works with minimal dependencies.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LangChainDocument:
    """Simple LangChain Document equivalent."""
    page_content: str
    metadata: Dict[str, Any]

@dataclass
class MultimodalDocument:
    """Multimodal document class."""
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

class SimpleEmbeddings:
    """Simple embeddings class compatible with LangChain interface."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self._embed_text(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Create a simple text embedding."""
        words = text.lower().split()
        vocab = set(words)
        
        features = []
        # Basic text statistics
        features.append(len(words))
        features.append(len(vocab))
        features.append(len(text))
        features.append(text.count('.'))
        features.append(text.count('?'))
        features.append(text.count('!'))
        
        # Character frequency features
        char_counts = {}
        for char in text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(char_counts.get(char, 0))
        
        # Word frequency features for common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        for word in common_words:
            features.append(text.lower().count(word))
        
        # Pad to consistent length
        while len(features) < 100:
            features.append(0.0)
        
        return features[:100]

class SimpleTextSplitter:
    """Simple text splitter similar to LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

class SimpleVectorStore:
    """Simple vector store with similarity search."""
    
    def __init__(self, embeddings: SimpleEmbeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents(self, documents: List[LangChainDocument]):
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        
        self.documents.extend(documents)
        self.vectors.extend(vectors)
    
    def similarity_search(self, query: str, k: int = 5) -> List[LangChainDocument]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        similarities = []
        
        for i, doc_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        results = []
        for i, similarity in similarities[:k]:
            doc = self.documents[i]
            # Add similarity score to metadata
            doc.metadata['similarity_score'] = similarity
            results.append(doc)
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class SimpleRetriever:
    """Simple retriever compatible with LangChain interface."""
    
    def __init__(self, vectorstore: SimpleVectorStore, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
        """Get relevant documents for a query."""
        return self.vectorstore.similarity_search(query, k=self.k)

class SimpleLLM:
    """Simple LLM for demonstration purposes."""
    
    def __call__(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        # This is a simple template-based response
        # In practice, you would integrate with a real LLM
        return f"""Based on the provided context, I can help answer your question. 

This is a demonstration response from the Simple LangChain RAG system. In a real implementation, this would be replaced with:
- OpenAI GPT models (via langchain.llms.OpenAI)
- Anthropic Claude (via langchain.llms.Anthropic)
- Local models via Ollama (via langchain.llms.Ollama)
- Hugging Face models (via langchain.llms.HuggingFacePipeline)

The system has successfully retrieved relevant context documents to help answer your query."""

class SimpleRetrievalQA:
    """Simple RetrievalQA chain."""
    
    def __init__(self, llm: SimpleLLM, retriever: SimpleRetriever):
        self.llm = llm
        self.retriever = retriever
    
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run the QA chain."""
        query = inputs.get("query", "")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        answer = self.llm(prompt)
        
        return {
            "query": query,
            "result": answer,
            "source_documents": docs
        }

class SimpleLangChainRAG:
    """Simple LangChain-compatible RAG system."""
    
    def __init__(self, collection_name: str = "simple_langchain_rag"):
        """Initialize the simple LangChain RAG system."""
        self.collection_name = collection_name
        
        # Initialize components
        self.embeddings = SimpleEmbeddings()
        self.text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = SimpleVectorStore(self.embeddings)
        self.retriever = SimpleRetriever(self.vectorstore, k=5)
        self.llm = SimpleLLM()
        self.qa_chain = SimpleRetrievalQA(self.llm, self.retriever)
        
        logger.info("Simple LangChain RAG system initialized successfully")
    
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
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate a simple image description."""
        width, height = image.size
        mode = image.mode
        
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            orientation = "wide landscape"
        elif aspect_ratio < 0.67:
            orientation = "tall portrait"
        else:
            orientation = "square"
        
        size_desc = "large" if width * height > 500000 else "medium" if width * height > 100000 else "small"
        
        return f"A {size_desc} {orientation} {mode.lower()} image ({width}x{height} pixels)"
    
    def add_documents(self, documents: List[MultimodalDocument]) -> bool:
        """Add documents to the RAG system."""
        try:
            langchain_docs = []
            
            for doc in documents:
                # Process text content
                text_content = doc.text
                
                # Add image information if available
                if doc.image_path or doc.image_url:
                    try:
                        image_source = doc.image_path or doc.image_url
                        image = self.load_image(image_source)
                        image_desc = self.describe_image(image)
                        text_content = f"{doc.text}\n\nImage Description: {image_desc}"
                    except Exception as e:
                        logger.warning(f"Could not process image for document {doc.id}: {e}")
                
                # Split text into chunks
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
            
            logger.info(f"Successfully added {len(documents)} documents ({len(langchain_docs)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def add_document(self, document: MultimodalDocument) -> bool:
        """Add a single document."""
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
                    "similarity_score": doc.metadata.get("similarity_score", 0.0)
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
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.vectorstore.documents),
            "vector_count": len(self.vectorstore.vectors),
            "embedding_dimension": len(self.vectorstore.vectors[0]) if self.vectorstore.vectors else 0,
            "system_type": "Simple LangChain Compatible"
        }
    
    def clear_collection(self) -> bool:
        """Clear all documents."""
        try:
            self.vectorstore.documents.clear()
            self.vectorstore.vectors.clear()
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

# Convenience functions
def create_simple_rag(collection_name: str = "simple_rag") -> SimpleLangChainRAG:
    """Create a simple RAG system."""
    return SimpleLangChainRAG(collection_name=collection_name)

def create_document(
    id: str,
    text: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    **metadata
) -> MultimodalDocument:
    """Create a multimodal document."""
    return MultimodalDocument(
        id=id,
        text=text,
        image_path=image_path,
        image_url=image_url,
        metadata=metadata
    )

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Simple LangChain RAG System")
    
    # Create RAG system
    rag = create_simple_rag("demo")
    
    # Create sample documents
    docs = [
        create_document(
            id="langchain_1",
            text="LangChain is a framework for developing applications powered by language models. It provides components like chains, agents, and memory.",
            category="Technology",
            topic="LangChain"
        ),
        create_document(
            id="rag_1",
            text="Retrieval-Augmented Generation (RAG) combines language models with external knowledge retrieval for more accurate and up-to-date responses.",
            category="Technology",
            topic="RAG"
        ),
        create_document(
            id="ai_1",
            text="Artificial Intelligence encompasses machine learning, deep learning, and various techniques for creating intelligent systems.",
            category="Technology",
            topic="AI"
        )
    ]
    
    # Add documents
    print("📚 Adding documents...")
    success = rag.add_documents(docs)
    print(f"Documents added: {success}")
    
    # Test search
    print("\n🔍 Testing search...")
    results = rag.search("What is LangChain?", k=2)
    print(f"Found {results['num_results']} results")
    
    for i, result in enumerate(results['results']):
        print(f"  {i+1}. Score: {result['similarity_score']:.3f}")
        print(f"     Content: {result['content'][:100]}...")
        print(f"     Topic: {result['metadata'].get('topic', 'N/A')}")
    
    # Test Q&A
    print("\n❓ Testing Q&A...")
    qa_result = rag.query("How does RAG work?")
    print(f"Question: {qa_result['question']}")
    print(f"Answer: {qa_result['answer'][:200]}...")
    print(f"Sources: {len(qa_result['source_documents'])}")
    
    # Get stats
    print("\n📊 Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Simple LangChain RAG system working!")
