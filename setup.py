#!/usr/bin/env python3
"""
Setup script for the Multimodal RAG System
Run this script to set up the environment and verify installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {description} failed:")
        print(f"     Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("     Please use Python 3.8 or higher")
        return False

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("  ❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    )

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ["./chroma_db", "./logs", "./data"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ Created directory: {directory}")
        except Exception as e:
            print(f"  ❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_installation():
    """Test if the installation is working."""
    print("🧪 Testing installation...")
    try:
        # Try importing the main module
        from multimodal_rag import MultimodalRAGProcessor, Document
        print("  ✅ Successfully imported MultimodalRAGProcessor")
        
        # Try creating a simple instance (this will test model loading)
        print("  🔄 Testing model loading (this may take a few minutes)...")
        processor = MultimodalRAGProcessor(
            collection_name="setup_test",
            persist_directory="./chroma_db"
        )
        print("  ✅ Models loaded successfully")
        
        # Test basic functionality
        doc = Document(
            id="setup_test_doc",
            text="This is a setup test document.",
            metadata={"test": True}
        )
        
        success = processor.add_document(doc)
        if success:
            print("  ✅ Document addition test passed")
        else:
            print("  ❌ Document addition test failed")
            return False
        
        results = processor.search("setup test")
        if results["num_results"] > 0:
            print("  ✅ Search test passed")
        else:
            print("  ❌ Search test failed")
            return False
        
        print("  🎉 Installation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("     Please check that all requirements are installed")
        return False
    except Exception as e:
        print(f"  ❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Multimodal RAG System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        print("❌ Failed to install requirements")
        print("💡 Try running manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed")
        print("💡 Try running: python test_system.py")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("  1. Run 'python example.py' to see the system in action")
    print("  2. Run 'python test_system.py' for comprehensive tests")
    print("  3. Check README.md for detailed usage instructions")
    print("  4. Customize config.py for your specific needs")
    print("\n💡 Tips:")
    print("  - The first run may take longer as models are downloaded")
    print("  - GPU acceleration will be used if available")
    print("  - Data persists in the ./chroma_db directory")

if __name__ == "__main__":
    main()
