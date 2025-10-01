"""
Query Processing - End-to-end RAG query implementation

Part of the RAG Production System course implementation.
Module 1: RAG Architecture & Core Implementation

License: MIT
"""

from typing import List, Dict, Any, Optional
import logging
import time
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

logger = logging.getLogger(__name__)


class RAGQueryProcessor:
    """
    End-to-end RAG query processor.

    Combines embedding generation, vector search, and LLM generation
    to provide comprehensive answers based on retrieved documents.
    """

    def __init__(self, embedding_generator, vector_store, llm_client=None):
        """
        Initialize the RAG query processor.

        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            llm_client: OpenAI client for LLM generation
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self._llm_client = llm_client

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                import openai
                self._llm_client = openai.OpenAI()
            except ImportError:
                raise ImportError("OpenAI library is required. Install with: pip install openai")
        return self._llm_client

    def query(
        self,
        question: str,
        top_k: int = 5,
        model: str = "gpt-4.1",
        max_context_length: int = 8000
    ) -> Dict[str, Any]:
        """
        Process a complete RAG query.

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            model: LLM model to use for generation
            max_context_length: Maximum context length in characters

        Returns:
            Dictionary containing answer, sources, and metadata

        Raises:
            ValueError: If question is empty
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        start_time = time.time()

        try:
            # Step 1: Generate query embedding
            logger.info(f"Processing query: {question[:100]}...")
            embedding_start = time.time()
            query_embedding = self.embedding_generator.embed_query(question)
            embedding_time = time.time() - embedding_start

            # Step 2: Search for relevant documents
            search_start = time.time()
            search_results = self.vector_store.search(query_embedding, top_k=top_k)
            search_time = time.time() - search_start

            if not search_results:
                logger.warning("No relevant documents found for query")
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'metadata': {
                        'query_time': time.time() - start_time,
                        'num_sources': 0
                    }
                }

            # Step 3: Build context from results
            context = self._build_context(search_results, max_context_length)

            # Step 4: Generate answer
            generation_start = time.time()
            answer = self._generate_answer(question, context, model)
            generation_time = time.time() - generation_start

            # Step 5: Format response
            total_time = time.time() - start_time

            response = {
                'answer': answer,
                'sources': self._format_sources(search_results),
                'confidence': self._calculate_confidence(search_results),
                'metadata': {
                    'query_time': total_time,
                    'embedding_time': embedding_time,
                    'search_time': search_time,
                    'generation_time': generation_time,
                    'num_sources': len(search_results),
                    'model': model
                }
            }

            logger.info(f"Query completed successfully in {total_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _build_context(self, search_results: List[Dict[str, Any]], max_length: int) -> str:
        """
        Build context string from search results.

        Args:
            search_results: List of search result dictionaries
            max_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            content = result.get('content', '').strip()
            if not content:
                continue

            # Format source with metadata
            source_info = f"Source {i+1}"
            if 'metadata' in result and 'source' in result['metadata']:
                source_info += f" ({result['metadata']['source']})"

            formatted_content = f"{source_info}:\n{content}\n"

            # Check if adding this content would exceed max length
            if current_length + len(formatted_content) > max_length:
                # Try to add a truncated version
                remaining_space = max_length - current_length - len(f"{source_info}:\n...\n")
                if remaining_space > 100:  # Only if we have reasonable space left
                    truncated_content = content[:remaining_space] + "..."
                    formatted_content = f"{source_info}:\n{truncated_content}\n"
                    context_parts.append(formatted_content)
                break

            context_parts.append(formatted_content)
            current_length += len(formatted_content)

        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str, model: str) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context: Retrieved and formatted context
            model: LLM model to use

        Returns:
            Generated answer string
        """
        prompt = self._build_prompt(question, context)

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. "
                                 "Always base your answers on the given context. If the context doesn't contain "
                                 "enough information to answer the question, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating an answer: {str(e)}"

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for LLM generation.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing.

Answer:"""

    def _format_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results as sources for the response.

        Args:
            search_results: Raw search results

        Returns:
            Formatted source information
        """
        sources = []

        for i, result in enumerate(search_results):
            source = {
                'id': result.get('id', f'source_{i}'),
                'score': round(result.get('score', 0.0), 3),
                'preview': result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', ''),
            }

            # Add metadata if available
            if 'metadata' in result:
                metadata = result['metadata']
                source.update({
                    'source_file': metadata.get('source', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', 0)
                })

            sources.append(source)

        return sources

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on search results.

        Args:
            search_results: List of search results

        Returns:
            Confidence score between 0 and 1
        """
        if not search_results:
            return 0.0

        # Simple confidence calculation based on top score
        top_score = search_results[0].get('score', 0.0)

        # Normalize score to confidence (this is a simple heuristic)
        if top_score > 0.9:
            return 0.95
        elif top_score > 0.8:
            return 0.85
        elif top_score > 0.7:
            return 0.75
        elif top_score > 0.6:
            return 0.65
        else:
            return 0.5


def rag_query(
    question: str,
    embedding_generator,
    vector_store,
    top_k: int = 3,
    model: str = "gpt-4.1"
) -> str:
    """
    Simple RAG query function for backwards compatibility.

    Args:
        question: User's question
        embedding_generator: EmbeddingGenerator instance
        vector_store: VectorStore instance
        top_k: Number of documents to retrieve
        model: LLM model to use

    Returns:
        Generated answer string
    """
    processor = RAGQueryProcessor(embedding_generator, vector_store)
    result = processor.query(question, top_k=top_k, model=model)
    return result['answer']