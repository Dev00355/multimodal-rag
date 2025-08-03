"""
Configuration settings for the Multimodal RAG System
Modify these settings to customize the behavior of your RAG system.
"""

# Model Configuration
MODELS = {
    # Text embedding model - used for semantic search
    "text_embedding": {
        "name": "all-MiniLM-L6-v2",
        "alternatives": [
            "all-mpnet-base-v2",  # Better quality, slower
            "all-distilroberta-v1",  # Good balance
            "paraphrase-MiniLM-L6-v2"  # Faster, smaller
        ]
    },
    
    # Vision-language model for image-text understanding
    "vision_language": {
        "name": "openai/clip-vit-base-patch32",
        "alternatives": [
            "openai/clip-vit-large-patch14",  # Better quality, larger
            "openai/clip-vit-base-patch16"   # Alternative base model
        ]
    },
    
    # Image captioning model
    "image_captioning": {
        "name": "Salesforce/blip-image-captioning-base",
        "alternatives": [
            "Salesforce/blip-image-captioning-large",  # Better quality
            "microsoft/git-base-coco"  # Alternative approach
        ]
    }
}

# Database Configuration
DATABASE = {
    "default_collection_name": "multimodal_rag",
    "persist_directory": "./chroma_db",
    "similarity_metric": "cosine",  # Options: cosine, l2, ip (inner product)
    "anonymized_telemetry": False
}

# Search Configuration
SEARCH = {
    "default_n_results": 5,
    "max_results": 50,
    "include_metadata_by_default": True,
    "similarity_threshold": 0.7  # Minimum similarity score for results
}

# Image Processing Configuration
IMAGE_PROCESSING = {
    "max_image_size": (1024, 1024),  # Maximum image dimensions
    "image_format": "RGB",
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "max_caption_length": 50,
    "timeout_seconds": 30  # Timeout for downloading images from URLs
}

# Logging Configuration
LOGGING = {
    "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": False,
    "log_file_path": "./multimodal_rag.log"
}

# Performance Configuration
PERFORMANCE = {
    "batch_size": 32,  # Batch size for processing multiple documents
    "use_gpu": True,   # Use GPU if available
    "num_threads": 4,  # Number of threads for parallel processing
    "cache_embeddings": True,  # Cache embeddings to avoid recomputation
    "memory_limit_gb": 8  # Maximum memory usage in GB
}

# Advanced Configuration
ADVANCED = {
    "text_chunk_size": 1000,  # Size of text chunks for long documents
    "text_chunk_overlap": 200,  # Overlap between text chunks
    "embedding_dimension": 384,  # Dimension of text embeddings
    "vision_embedding_dimension": 512,  # Dimension of vision embeddings
    "normalize_embeddings": True,  # Normalize embeddings for better similarity
    "use_mixed_precision": True  # Use mixed precision for faster inference
}

# Environment-specific settings
ENVIRONMENT = {
    "development": {
        "log_level": "DEBUG",
        "cache_models": False,
        "validate_inputs": True
    },
    "production": {
        "log_level": "WARNING",
        "cache_models": True,
        "validate_inputs": False
    }
}

# Default configuration function
def get_config(environment: str = "development") -> dict:
    """
    Get configuration dictionary for the specified environment.
    
    Args:
        environment: Environment name ("development" or "production")
    
    Returns:
        Dictionary containing all configuration settings
    """
    config = {
        "models": MODELS,
        "database": DATABASE,
        "search": SEARCH,
        "image_processing": IMAGE_PROCESSING,
        "logging": LOGGING,
        "performance": PERFORMANCE,
        "advanced": ADVANCED
    }
    
    # Apply environment-specific overrides
    if environment in ENVIRONMENT:
        env_config = ENVIRONMENT[environment]
        config["logging"]["level"] = env_config.get("log_level", config["logging"]["level"])
        config["performance"]["cache_models"] = env_config.get("cache_models", config["performance"].get("cache_models", True))
        config["advanced"]["validate_inputs"] = env_config.get("validate_inputs", config["advanced"].get("validate_inputs", True))
    
    return config

# Quick access to commonly used settings
def get_model_names():
    """Get the default model names."""
    return {
        "text_embedding": MODELS["text_embedding"]["name"],
        "vision_language": MODELS["vision_language"]["name"],
        "image_captioning": MODELS["image_captioning"]["name"]
    }

def get_database_settings():
    """Get the default database settings."""
    return DATABASE

def get_search_settings():
    """Get the default search settings."""
    return SEARCH
