#!/usr/bin/env python3
"""
LangChain Multimodal RAG Example
Demonstrates the LangChain-compatible multimodal RAG system.
"""

from langchain_multimodal_rag import (
    LangChainMultimodalRAG,
    MultimodalDocument,
    create_multimodal_rag,
    create_document
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 LangChain Multimodal RAG System Demo")
    print("=" * 50)
    
    # Create the RAG system
    print("🔧 Initializing LangChain Multimodal RAG...")
    rag = create_multimodal_rag(
        collection_name="langchain_demo",
        persist_directory="./langchain_demo_db"
    )
    print("✅ System initialized!")
    
    # Create sample documents with various content types
    print("\n📚 Creating sample documents...")
    
    documents = [
        create_document(
            id="ai_overview",
            text="""
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            This includes learning, reasoning, problem-solving, perception, and language understanding.
            
            Machine Learning is a subset of AI that focuses on the development of algorithms and 
            statistical models that enable computers to improve their performance on a specific 
            task through experience, without being explicitly programmed.
            
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in data.
            """,
            category="Technology",
            topic="Artificial Intelligence",
            difficulty="Beginner",
            author="AI Researcher"
        ),
        
        create_document(
            id="langchain_intro",
            text="""
            LangChain is a framework for developing applications powered by language models. 
            It provides a set of tools, components, and interfaces that simplify the process 
            of creating LLM-powered applications.
            
            Key components of LangChain include:
            - LLMs and Chat Models: Wrappers around language models
            - Prompts: Templates for formatting inputs to language models
            - Chains: Sequences of calls to LLMs or other utilities
            - Agents: LLMs that make decisions about which actions to take
            - Memory: Ways to persist state between calls of a chain/agent
            - Document Loaders: Utilities for loading data from various sources
            - Vector Stores: Databases for storing and searching over unstructured data
            """,
            category="Technology",
            topic="LangChain",
            difficulty="Intermediate",
            author="LangChain Developer"
        ),
        
        create_document(
            id="rag_explanation",
            text="""
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
            large language models with external knowledge retrieval. Instead of relying solely 
            on the model's training data, RAG systems can access and incorporate relevant 
            information from external sources in real-time.
            
            The RAG process typically involves:
            1. Indexing: Documents are processed and stored in a vector database
            2. Retrieval: Relevant documents are found based on the user's query
            3. Generation: The language model generates a response using both the query 
               and the retrieved context
            
            Benefits of RAG:
            - Access to up-to-date information
            - Reduced hallucinations
            - Ability to cite sources
            - Domain-specific knowledge integration
            """,
            category="Technology",
            topic="RAG",
            difficulty="Intermediate",
            author="ML Engineer"
        ),
        
        create_document(
            id="multimodal_ai",
            text="""
            Multimodal AI refers to artificial intelligence systems that can process and 
            understand multiple types of data simultaneously, such as text, images, audio, 
            and video. This approach mimics human perception, which naturally integrates 
            information from multiple senses.
            
            Key technologies in multimodal AI:
            - CLIP (Contrastive Language-Image Pre-training): Connects text and images
            - BLIP (Bootstrapping Language-Image Pre-training): Generates image captions
            - Vision Transformers: Apply transformer architecture to image processing
            - Multimodal embeddings: Unified representations for different data types
            
            Applications include:
            - Image captioning and description
            - Visual question answering
            - Content-based image retrieval
            - Multimodal search engines
            """,
            category="Technology",
            topic="Multimodal AI",
            difficulty="Advanced",
            author="Computer Vision Researcher"
        )
    ]
    
    # Add documents to the system
    print(f"📝 Adding {len(documents)} documents to the system...")
    success = rag.add_documents(documents)
    
    if success:
        print("✅ All documents added successfully!")
    else:
        print("❌ Error adding documents")
        return
    
    # Get system statistics
    print("\n📊 System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate search functionality
    print("\n🔍 Search Functionality Demo")
    print("-" * 30)
    
    search_queries = [
        "What is machine learning?",
        "How does LangChain work?",
        "Explain RAG systems",
        "multimodal AI applications"
    ]
    
    for query in search_queries:
        print(f"\n🔎 Query: '{query}'")
        results = rag.search(query, k=2)
        
        print(f"Found {results['num_results']} relevant documents:")
        for i, result in enumerate(results['results']):
            metadata = result['metadata']
            print(f"  {i+1}. Document: {metadata.get('original_id', 'Unknown')}")
            print(f"     Topic: {metadata.get('topic', 'N/A')}")
            print(f"     Difficulty: {metadata.get('difficulty', 'N/A')}")
            print(f"     Content: {result['content'][:150]}...")
            print()
    
    # Demonstrate Q&A functionality
    print("\n❓ Question & Answer Demo")
    print("-" * 25)
    
    qa_questions = [
        "What are the main components of LangChain?",
        "How does RAG improve language model responses?",
        "What technologies are used in multimodal AI?",
        "What is the difference between machine learning and deep learning?"
    ]
    
    for question in qa_questions:
        print(f"\n❓ Question: {question}")
        result = rag.query(question)
        
        print(f"🤖 Answer: {result['answer']}")
        print(f"📚 Sources: {len(result['source_documents'])} documents")
        
        # Show source information
        for i, source in enumerate(result['source_documents'][:2]):  # Show first 2 sources
            metadata = source['metadata']
            print(f"   Source {i+1}: {metadata.get('original_id', 'Unknown')} "
                  f"({metadata.get('topic', 'N/A')})")
        print()
    
    # Demonstrate LangChain integration features
    print("\n🔗 LangChain Integration Features")
    print("-" * 35)
    
    print("✅ Document chunking with RecursiveCharacterTextSplitter")
    print("✅ Vector storage with Chroma")
    print("✅ Custom multimodal embeddings")
    print("✅ LangChain-compatible retriever")
    print("✅ RetrievalQA chain integration")
    print("✅ Metadata preservation and filtering")
    print("✅ Persistent storage")
    
    # Show advanced usage patterns
    print("\n🚀 Advanced Usage Patterns")
    print("-" * 25)
    
    print("1. Custom Retriever:")
    print("   retriever = rag.retriever")
    print("   docs = retriever.get_relevant_documents('your query')")
    
    print("\n2. Direct Vector Store Access:")
    print("   vectorstore = rag.vectorstore")
    print("   docs = vectorstore.similarity_search('query', k=5)")
    
    print("\n3. Chain Customization:")
    print("   from langchain.chains import RetrievalQA")
    print("   custom_chain = RetrievalQA.from_chain_type(")
    print("       llm=your_llm, retriever=rag.retriever)")
    
    print("\n4. Metadata Filtering:")
    print("   docs = vectorstore.similarity_search(")
    print("       'query', filter={'category': 'Technology'})")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Next Steps:")
    print("  - Integrate with your preferred LLM (OpenAI, Anthropic, Ollama)")
    print("  - Add image documents with image_path or image_url")
    print("  - Customize the embedding models for your domain")
    print("  - Implement custom chains and agents")
    print("  - Add metadata filtering for specific use cases")

if __name__ == "__main__":
    main()
