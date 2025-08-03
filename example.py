#!/usr/bin/env python3
"""
Example usage of the Multimodal RAG System
This script demonstrates how to use the MultimodalRAGProcessor
"""

from multimodal_rag import MultimodalRAGProcessor, Document
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Initializing Multimodal RAG System...")
    
    # Initialize the processor
    processor = MultimodalRAGProcessor(
        collection_name="example_collection",
        persist_directory="./example_db"
    )
    
    print("✅ System initialized successfully!")
    
    # Create some example documents
    documents = [
        Document(
            id="ai_doc_1",
            text="Artificial Intelligence is transforming how we work and live. Machine learning algorithms can now process vast amounts of data to find patterns and make predictions.",
            metadata={"category": "Technology", "topic": "AI", "difficulty": "beginner"}
        ),
        Document(
            id="ml_doc_1",
            text="Deep learning neural networks have revolutionized computer vision and natural language processing. These models can learn complex representations from raw data.",
            metadata={"category": "Technology", "topic": "Machine Learning", "difficulty": "intermediate"}
        ),
        Document(
            id="nature_doc_1",
            text="The Amazon rainforest is home to incredible biodiversity. It contains thousands of species of plants, animals, and insects that are found nowhere else on Earth.",
            metadata={"category": "Nature", "topic": "Environment", "difficulty": "beginner"}
        ),
        Document(
            id="space_doc_1",
            text="The James Webb Space Telescope has captured stunning images of distant galaxies, revealing the early universe in unprecedented detail. These observations help us understand cosmic evolution.",
            metadata={"category": "Science", "topic": "Astronomy", "difficulty": "intermediate"}
        )
    ]
    
    print(f"📚 Adding {len(documents)} documents to the system...")
    
    # Add documents to the system
    for doc in documents:
        success = processor.add_document(doc)
        if success:
            print(f"  ✅ Added document: {doc.id}")
        else:
            print(f"  ❌ Failed to add document: {doc.id}")
    
    # Get collection statistics
    stats = processor.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"  Collection Name: {stats['collection_name']}")
    print(f"  Total Documents: {stats['document_count']}")
    print(f"  Storage Directory: {stats['persist_directory']}")
    
    # Perform some example searches
    queries = [
        "artificial intelligence and machine learning",
        "space telescope and galaxies",
        "rainforest biodiversity",
        "neural networks and deep learning"
    ]
    
    print(f"\n🔍 Performing example searches...")
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        results = processor.search(query, n_results=3)
        
        if results['num_results'] > 0:
            print(f"  Found {results['num_results']} relevant documents:")
            
            for i, doc_content in enumerate(results['results']['documents'][0]):
                metadata = results['results']['metadatas'][0][i]
                distance = results['results']['distances'][0][i]
                similarity = 1 - distance
                
                print(f"    {i+1}. Similarity: {similarity:.3f}")
                print(f"       Category: {metadata.get('category', 'N/A')}")
                print(f"       Topic: {metadata.get('topic', 'N/A')}")
                print(f"       Content: {doc_content[:100]}...")
                print()
        else:
            print("  No relevant documents found.")
    
    print("🎉 Example completed successfully!")
    print("\n💡 Tips for using the system:")
    print("  - Add documents with images using image_path or image_url")
    print("  - Use descriptive metadata to organize your documents")
    print("  - Experiment with different query styles")
    print("  - The system automatically generates captions for images")
    print("  - Data persists between runs in the ChromaDB directory")

if __name__ == "__main__":
    main()
