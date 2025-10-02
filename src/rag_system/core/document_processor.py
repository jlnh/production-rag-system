"""
Document Processor - Multi-format document processing for RAG systems

Part of the RAG Production System course implementation.
Module 1: RAG Architecture & Core Implementation

License: MIT
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Multi-format document processor for RAG systems.

    Handles PDF, DOCX, and other document formats, converting them into
    text chunks suitable for embedding and retrieval.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a document file and return text chunks.

        Args:
            file_path: Path to the document file

        Returns:
            List of dictionaries containing chunk content and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        try:
            if file_ext == ".pdf":
                text = self.parse_pdf(file_path)
            elif file_ext == ".docx":
                text = self.parse_docx(file_path)
            elif file_ext in [".txt", ".md"]:
                text = file_path.read_text(encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            logger.info(f"Successfully processed {file_path}, extracted {len(text)} characters")
            return self.create_chunks(text, str(file_path))

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def parse_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(
                            f"Error extracting page {page_num} from {file_path}: {str(e)}"
                        )
                        continue

                return text.strip()

        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF processing. Install with: pip install PyPDF2"
            )
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise

    def parse_docx(self, file_path: Path) -> str:
        """
        Extract text from DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            Extracted text content
        """
        try:
            import docx

            doc = docx.Document(file_path)
            text = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)

            return "\n".join(text)

        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX processing. Install with: pip install python-docx"
            )
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            raise

    def create_chunks(self, text: str, source_path: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for processing.

        Args:
            text: Source text to chunk
            source_path: Path to source file for metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text.strip():
            logger.warning(f"Empty text provided for chunking from {source_path}")
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_content = " ".join(chunk_words)

            chunk = {
                "content": chunk_content,
                "metadata": {
                    "source": source_path,
                    "chunk_id": i // (self.chunk_size - self.overlap),
                    "start_word": i,
                    "end_word": i + len(chunk_words),
                    "word_count": len(chunk_words),
                },
            }
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from {source_path}")
        return chunks
