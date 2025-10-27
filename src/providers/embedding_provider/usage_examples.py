"""
Embedding Provider Usage Examples

This file demonstrates various usage patterns for the embedding provider system.
Copy these examples into your application code as needed.
"""

import asyncio
from typing import List

from src.providers.embedding_provider import (
    create_embedding_provider,
    get_default_embedding_provider,
    EmbeddingConfig,
    EmbeddingTaskType,
    EmbeddingProviderError,
)


# ============================================================================
# EXAMPLE 1: Basic Single Text Embedding (Async)
# ============================================================================


async def example_basic_embedding_async():
    """Generate embedding for a single text asynchronously"""

    # Create provider with defaults
    provider = create_embedding_provider(
        provider_type="openai", model_name="text-embedding-3-small"
    )

    # Generate embedding
    text = "Machine learning is a subset of artificial intelligence."
    embedding = await provider.generate_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


# ============================================================================
# EXAMPLE 2: Basic Single Text Embedding (Sync)
# ============================================================================


def example_basic_embedding_sync():
    """Generate embedding for a single text synchronously"""

    # Create provider
    provider = create_embedding_provider(
        provider_type="gemini", model_name="text-embedding-004"
    )

    # Generate embedding (synchronous)
    text = "Natural language processing enables computers to understand text."
    embedding = provider.generate_embedding_sync(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


# ============================================================================
# EXAMPLE 3: Batch Embedding Generation (Async)
# ============================================================================


async def example_batch_embeddings():
    """Generate embeddings for multiple texts efficiently"""

    # Create provider
    provider = create_embedding_provider(
        provider_type="openai", model_name="text-embedding-3-small"
    )

    # List of texts to embed
    texts = [
        "Python is a programming language.",
        "JavaScript is used for web development.",
        "SQL is for database queries.",
        "Machine learning uses algorithms to learn patterns.",
        "Deep learning is a subset of machine learning.",
    ]

    # Generate batch embeddings
    embeddings = await provider.generate_embeddings_batch(texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")

    # Use embeddings (e.g., for similarity search)
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"{i+1}. {text[:50]}... -> {embedding[:3]}...")


# ============================================================================
# EXAMPLE 4: Task-Specific Embeddings
# ============================================================================


async def example_task_specific_embeddings():
    """Use different task types for optimized embeddings"""

    # For semantic similarity
    similarity_provider = create_embedding_provider(
        provider_type="gemini",
        model_name="text-embedding-004",
        task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY,
    )

    # For retrieval (query)
    query_provider = create_embedding_provider(
        provider_type="gemini",
        model_name="text-embedding-004",
        task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
    )

    # For retrieval (document)
    document_provider = create_embedding_provider(
        provider_type="gemini",
        model_name="text-embedding-004",
        task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
    )

    text = "What is machine learning?"

    # Different embeddings for different tasks
    similarity_emb = await similarity_provider.generate_embedding(text)
    query_emb = await query_provider.generate_embedding(text)
    document_emb = await document_provider.generate_embedding(text)

    print(f"Similarity embedding: {similarity_emb[:3]}...")
    print(f"Query embedding: {query_emb[:3]}...")
    print(f"Document embedding: {document_emb[:3]}...")


# ============================================================================
# EXAMPLE 5: Advanced Configuration with EmbeddingConfig
# ============================================================================


async def example_advanced_config():
    """Using EmbeddingConfig for detailed configuration"""

    # Create detailed configuration
    config = EmbeddingConfig(
        model_name="text-embedding-3-large",
        embedding_dimension=3072,
        task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY,
        batch_size=50,
        timeout=120,
        max_retries=3,
        strip_whitespace=True,
        remove_empty=True,
        max_text_length=8000,
    )

    # Create provider with config
    provider = create_embedding_provider(provider_type="openai", config=config)

    # Use provider
    text = "  This text has extra whitespace  "
    embedding = await provider.generate_embedding(text)

    print(f"Original text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Configuration: {provider.config.to_dict()}")


# ============================================================================
# EXAMPLE 6: Dynamic Configuration Updates
# ============================================================================


async def example_dynamic_config():
    """Updating provider configuration at runtime"""

    # Create provider
    provider = create_embedding_provider(provider_type="openai", batch_size=10)

    # Initial batch with size 10
    texts_small = [f"Text {i}" for i in range(10)]
    embeddings1 = await provider.generate_embeddings_batch(texts_small)
    print(f"Processed {len(embeddings1)} embeddings with batch_size=10")

    # Update configuration dynamically
    provider.update_config(batch_size=50)

    # Larger batch with size 50
    texts_large = [f"Text {i}" for i in range(50)]
    embeddings2 = await provider.generate_embeddings_batch(texts_large)
    print(f"Processed {len(embeddings2)} embeddings with batch_size=50")


# ============================================================================
# EXAMPLE 7: Semantic Similarity Search
# ============================================================================


async def example_semantic_similarity():
    """Find similar texts using embeddings"""

    import numpy as np

    # Create provider
    provider = create_embedding_provider(
        provider_type="openai", task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY
    )

    # Documents to search
    documents = [
        "Python is a high-level programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "JavaScript is commonly used for web development.",
        "Mount Everest is the tallest mountain on Earth.",
        "Java is an object-oriented programming language.",
    ]

    # Query
    query = "Tell me about programming languages"

    # Generate embeddings
    doc_embeddings = await provider.generate_embeddings_batch(documents)
    query_embedding = await provider.generate_embedding(query)

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find most similar documents
    similarities = [
        cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
    ]

    # Sort by similarity
    ranked_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("\nMost similar documents:")
    for i, (doc, score) in enumerate(ranked_docs[:3], 1):
        print(f"{i}. {doc} (similarity: {score:.4f})")


# ============================================================================
# EXAMPLE 8: Comparing OpenAI and Gemini Providers
# ============================================================================


async def example_compare_providers():
    """Compare embeddings from different providers"""

    # Create both providers
    openai_provider = create_embedding_provider(
        provider_type="openai", model_name="text-embedding-3-small"
    )

    gemini_provider = create_embedding_provider(
        provider_type="gemini", model_name="text-embedding-004"
    )

    text = "Artificial intelligence is transforming technology."

    # Generate embeddings from both
    openai_emb = await openai_provider.generate_embedding(text)
    gemini_emb = await gemini_provider.generate_embedding(text)

    print(f"Text: {text}")
    print(f"OpenAI embedding dimension: {len(openai_emb)}")
    print(f"Gemini embedding dimension: {len(gemini_emb)}")
    print(f"OpenAI first 3 values: {openai_emb[:3]}")
    print(f"Gemini first 3 values: {gemini_emb[:3]}")


# ============================================================================
# EXAMPLE 9: Error Handling
# ============================================================================


async def example_error_handling():
    """Proper error handling for embedding operations"""

    try:
        # Create provider
        provider = create_embedding_provider(
            provider_type="openai", model_name="text-embedding-3-small"
        )

        # Attempt to embed empty text
        empty_text = "   "  # Only whitespace

        try:
            embedding = await provider.generate_embedding(empty_text)
            print(f"Unexpected success: {len(embedding)} dimensions")
        except EmbeddingProviderError:
            # This is expected - empty text should raise an error
            pass

    except EmbeddingProviderError as e:
        print(f"Embedding error occurred: {e}")
        print(f"Error details: {e.details}")
        # Handle specific embedding errors
        # Could implement retry logic, use fallback text, etc.

    except Exception as e:
        print(f"Unexpected error: {e}")


# ============================================================================
# EXAMPLE 10: Large Batch Processing with Progress
# ============================================================================


async def example_large_batch_processing():
    """Process large batches with progress tracking"""

    provider = create_embedding_provider(provider_type="openai", batch_size=100)

    # Simulate large dataset
    large_dataset = [
        f"Document number {i} about various topics in AI and ML." for i in range(500)
    ]

    print(f"Processing {len(large_dataset)} documents...")

    # Process in chunks with progress
    all_embeddings = []
    chunk_size = 100

    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset[i : i + chunk_size]
        embeddings = await provider.generate_embeddings_batch(chunk)
        all_embeddings.extend(embeddings)

        progress = (i + len(chunk)) / len(large_dataset) * 100
        print(f"Progress: {progress:.1f}% ({i + len(chunk)}/{len(large_dataset)})")

    print(f"Completed! Generated {len(all_embeddings)} embeddings")


# ============================================================================
# EXAMPLE 11: Provider Information and Monitoring
# ============================================================================


async def example_provider_info():
    """Getting information about provider configuration"""

    provider = create_embedding_provider(
        provider_type="gemini",
        model_name="text-embedding-004",
        task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        batch_size=50,
    )

    # Get provider information
    info = provider.get_model_info()
    print("Provider Information:")
    print(f"  Type: {info['provider_type']}")
    print(f"  Model: {info['model_name']}")
    print(f"  Dimension: {info['embedding_dimension']}")
    print(f"  Task Type: {info['task_type']}")
    print(f"  Batch Size: {info['batch_size']}")
    print(f"  Configuration: {info['config']}")

    # Estimate tokens for text
    test_text = "This is a sample text for token estimation."
    estimated_tokens = provider.estimate_tokens(test_text)
    print(f"\nToken estimation for '{test_text}':")
    print(f"  Estimated tokens: {estimated_tokens}")


# ============================================================================
# EXAMPLE 12: Using Default Provider
# ============================================================================


async def example_default_provider():
    """Using the default provider from application settings"""

    # Get default provider (uses settings from config)
    provider = get_default_embedding_provider()

    # Use with defaults
    text = "Using default embedding provider configuration."
    embedding = await provider.generate_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Provider info: {provider.get_model_info()}")


# ============================================================================
# EXAMPLE 13: Building a Vector Database
# ============================================================================


async def example_vector_database():
    """Build a simple in-memory vector database"""

    import numpy as np
    from typing import Tuple

    class SimpleVectorDB:
        def __init__(self, provider):
            self.provider = provider
            self.documents = []
            self.embeddings = []

        async def add_documents(self, docs: List[str]):
            """Add documents to the database"""
            embeddings = await self.provider.generate_embeddings_batch(docs)
            self.documents.extend(docs)
            self.embeddings.extend(embeddings)
            print(f"Added {len(docs)} documents. Total: {len(self.documents)}")

        async def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
            """Search for most similar documents"""
            query_emb = await self.provider.generate_embedding(query)

            # Calculate similarities
            similarities = []
            for doc, doc_emb in zip(self.documents, self.embeddings):
                sim = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                similarities.append((doc, sim))

            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    # Create database
    provider = create_embedding_provider(
        provider_type="openai", task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT
    )

    db = SimpleVectorDB(provider)

    # Add documents
    documents = [
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning models can predict outcomes based on historical data patterns.",
        "The Renaissance was a period of cultural rebirth in Europe during the 14th-17th centuries.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "Quantum computing uses quantum bits that can exist in multiple states simultaneously.",
    ]

    await db.add_documents(documents)

    # Search
    query = "How do plants make energy?"
    results = await db.search(query, top_k=3)

    print(f"\nQuery: {query}")
    print("\nTop results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")


# ============================================================================
# EXAMPLE 14: Multi-Provider Ensemble
# ============================================================================


async def example_multi_provider_ensemble():
    """Use multiple providers for ensemble embeddings"""

    import numpy as np

    # Create multiple providers
    providers = {
        "openai": create_embedding_provider(
            provider_type="openai", model_name="text-embedding-3-small"
        ),
        "gemini": create_embedding_provider(
            provider_type="gemini", model_name="text-embedding-004"
        ),
    }

    text = "Ensemble methods combine multiple models for better predictions."

    # Generate embeddings from all providers
    embeddings = {}
    for name, provider in providers.items():
        emb = await provider.generate_embedding(text)
        # Normalize to unit length for fair comparison
        emb_normalized = emb / np.linalg.norm(emb)
        embeddings[name] = emb_normalized

    print(f"Text: {text}")
    print("\nProvider embeddings:")
    for name, emb in embeddings.items():
        print(f"{name}: dimension={len(emb)}, first 3 values={emb[:3]}")


# ============================================================================
# EXAMPLE 15: Preprocessing and Validation
# ============================================================================


async def example_preprocessing():
    """Demonstrate text preprocessing and validation"""

    provider = create_embedding_provider(
        provider_type="openai", model_name="text-embedding-3-small"
    )

    # Texts with various issues
    texts = [
        "  Normal text with extra spaces  ",
        "",  # Empty
        "   ",  # Only whitespace
        "Valid text without issues",
    ]

    print("Processing texts with preprocessing:")
    for i, text in enumerate(texts):
        try:
            embedding = await provider.generate_embedding(text)
            print(f"{i+1}. Success: '{text}' -> {len(embedding)} dimensions")
        except EmbeddingProviderError as e:
            print(f"{i+1}. Failed: '{text}' -> {e}")


# ============================================================================
# Main execution
# ============================================================================


async def main():
    """Run selected examples"""

    print("=" * 80)
    print("EXAMPLE 1: Basic Embedding (Async)")
    print("=" * 80)
    await example_basic_embedding_async()

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Batch Embeddings")
    print("=" * 80)
    await example_batch_embeddings()

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Task-Specific Embeddings")
    print("=" * 80)
    await example_task_specific_embeddings()

    print("\n" + "=" * 80)
    print("EXAMPLE 7: Semantic Similarity Search")
    print("=" * 80)
    await example_semantic_similarity()

    print("\n" + "=" * 80)
    print("EXAMPLE 11: Provider Information")
    print("=" * 80)
    await example_provider_info()

    # Add more example calls as needed


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
