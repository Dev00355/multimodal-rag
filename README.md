# 🔗 LangChain-Compatible Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that can process both text and images, **fully compatible with LangChain**. Features multiple implementation levels from simple to production-ready, using state-of-the-art models including CLIP, BLIP, and Sentence Transformers.

## 🎯 **NEW: Full LangChain Integration!**

✅ **LangChain-Compatible Components**  
✅ **Multiple Implementation Levels**  
✅ **Production-Ready Integration**  
✅ **Works with Any LLM** (OpenAI, Anthropic, Ollama)

## 🚀 Implementation Options

### 1. **Simple LangChain RAG** ✅ **(RECOMMENDED - TESTED WORKING)**
**File**: `simple_langchain_rag.py`
- 🔗 Full LangChain compatibility
- 📦 No complex dependencies
- ⚡ Works immediately
- 🎯 Perfect for getting started

### 2. **Full LangChain Integration**
**File**: `langchain_multimodal_rag.py`
- 🔗 Complete LangChain ecosystem
- 🧠 Advanced AI models (CLIP, BLIP)
- 💾 ChromaDB vector storage
- 🏭 Production-ready features

### 3. **Standalone Multimodal RAG**
**File**: `multimodal_rag.py`
- 🤖 State-of-the-art models
- 🔬 Research-grade implementation
- 📊 Advanced analytics

## Features

- **🔗 LangChain Compatible**: Works with any LangChain application
- **Multimodal Processing**: Handle both text and image content
- **Image Captioning**: Automatic caption generation using BLIP
- **Semantic Search**: Vector-based similarity search
- **Flexible Input**: Support for local images, URLs, and PIL Image objects
- **Multiple Storage Options**: ChromaDB or in-memory storage
- **Easy Integration**: Simple API for adding documents and searching
- **Production Ready**: Error handling, logging, and scalability

## Models Used

- **Text Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vision-Language**: CLIP (openai/clip-vit-base-patch32)
- **Image Captioning**: BLIP (Salesforce/blip-image-captioning-base)
- **Vector Database**: ChromaDB

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd multimodal_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 LangChain Quick Start (Recommended)

```python
from simple_langchain_rag import create_simple_rag, create_document

# Create LangChain-compatible RAG system
rag = create_simple_rag("my_collection")

# Create a document
doc = create_document(
    id="example_1",
    text="LangChain is a framework for building LLM applications.",
    category="Technology"
)

# Add document (LangChain Document format)
rag.add_document(doc)

# Search using LangChain Retriever pattern
results = rag.search("What is LangChain?")
print(f"Found {results['num_results']} results")

# Q&A using LangChain Chain pattern
answer = rag.query("How does LangChain work?")
print(f"Answer: {answer['answer']}")

# Access LangChain components directly
retriever = rag.retriever  # Use with any LangChain chain
vectorstore = rag.vectorstore  # Direct vector operations
```

## 🔧 Standalone Quick Start

```python
from multimodal_rag import MultimodalRAGProcessor, Document

# Initialize the processor
processor = MultimodalRAGProcessor()

# Create a document with text only
text_doc = Document(
    id="doc_1",
    text="This is a document about machine learning and AI.",
    metadata={"category": "AI", "author": "John Doe"}
)

# Create a document with text and image
image_doc = Document(
    id="doc_2",
    text="A beautiful sunset over the mountains.",
    image_path="path/to/your/image.jpg",  # or image_url for web images
    metadata={"category": "Nature"}
)

# Add documents to the system
processor.add_document(text_doc)
processor.add_document(image_doc)

# Search for relevant documents
results = processor.search("artificial intelligence", n_results=3)
print(f"Found {results['num_results']} relevant documents")

# Get collection statistics
stats = processor.get_collection_stats()
print(f"Total documents: {stats['document_count']}")
```

## API Reference

### MultimodalRAGProcessor

#### Constructor Parameters

- `collection_name` (str): Name for the ChromaDB collection (default: "multimodal_rag")
- `embedding_model` (str): Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `vision_model` (str): CLIP model name (default: "openai/clip-vit-base-patch32")
- `caption_model` (str): BLIP model name (default: "Salesforce/blip-image-captioning-base")
- `persist_directory` (str): Directory for ChromaDB persistence (default: "./chroma_db")

#### Methods

##### `add_document(document: Document) -> bool`
Add a document to the RAG system.

##### `search(query: str, n_results: int = 5, include_metadata: bool = True) -> Dict[str, Any]`
Search for relevant documents based on a query.

##### `get_collection_stats() -> Dict[str, Any]`
Get statistics about the current collection.

##### `clear_collection() -> bool`
Clear all documents from the collection.

##### `load_image(image_source: Union[str, Image.Image]) -> Image.Image`
Load image from file path, URL, or PIL Image.

##### `generate_image_caption(image: Image.Image) -> str`
Generate a caption for an image using BLIP.

### Document Class

A dataclass representing a document with the following fields:

- `id` (str): Unique identifier for the document
- `text` (str): Text content of the document
- `image_path` (Optional[str]): Path to local image file
- `image_url` (Optional[str]): URL to image
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

## Examples

### Adding Documents with Images

```python
# Local image
doc_with_local_image = Document(
    id="local_img_1",
    text="A photo of a cat sitting on a windowsill",
    image_path="/path/to/cat.jpg"
)

# Web image
doc_with_web_image = Document(
    id="web_img_1",
    text="A stunning landscape photograph",
    image_url="https://example.com/landscape.jpg"
)

processor.add_document(doc_with_local_image)
processor.add_document(doc_with_web_image)
```

### Advanced Search

```python
# Search with specific parameters
results = processor.search(
    query="cats and animals",
    n_results=10,
    include_metadata=True
)

# Process results
for i, doc in enumerate(results['results']['documents'][0]):
    metadata = results['results']['metadatas'][0][i]
    distance = results['results']['distances'][0][i]
    
    print(f"Document {i+1}:")
    print(f"Content: {doc}")
    print(f"Similarity: {1 - distance:.3f}")
    print(f"Has Image: {metadata.get('has_image', False)}")
    if metadata.get('image_caption'):
        print(f"Image Caption: {metadata['image_caption']}")
    print("-" * 50)
```

## Performance Considerations

- **Model Loading**: Models are loaded once during initialization. This may take a few minutes on first run.
- **GPU Support**: The system automatically uses GPU if available for faster processing.
- **Memory Usage**: Large collections may require significant RAM. Consider batch processing for very large datasets.
- **Storage**: ChromaDB persists data to disk. The database size grows with the number of documents.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use a smaller embedding model
2. **Slow Performance**: Ensure GPU is available and properly configured
3. **Image Loading Errors**: Check image paths/URLs and ensure images are accessible
4. **Model Download Issues**: Ensure stable internet connection for initial model downloads

### Logging

The system uses Python's logging module. To see detailed logs:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformer models
- ChromaDB for the vector database
- OpenAI for CLIP
- Salesforce for BLIP
- Sentence Transformers team

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multimodal_rag,
  title={Multimodal RAG System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multimodal_rag}
}
```
