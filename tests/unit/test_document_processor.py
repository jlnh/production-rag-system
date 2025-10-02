"""
Unit Tests for Document Processor

Tests the document processing functionality including:
- File format support
- Text chunking
- Metadata handling
- Error cases
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, Mock

from rag_system.core import DocumentProcessor


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 500
        assert processor.overlap == 50

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        processor = DocumentProcessor(chunk_size=1000, overlap=100)
        assert processor.chunk_size == 1000
        assert processor.overlap == 100

    def test_create_chunks_basic(self, document_processor):
        """Test basic text chunking functionality."""
        text = "This is a test document. " * 50  # 250 words
        chunks = document_processor.create_chunks(text, "test.txt")

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

        # Check first chunk
        first_chunk = chunks[0]
        assert first_chunk["metadata"]["source"] == "test.txt"
        assert first_chunk["metadata"]["chunk_id"] == 0
        assert "start_word" in first_chunk["metadata"]
        assert "end_word" in first_chunk["metadata"]
        assert "word_count" in first_chunk["metadata"]

    def test_create_chunks_overlap(self):
        """Test that chunks have proper overlap."""
        processor = DocumentProcessor(chunk_size=10, overlap=3)
        text = "word " * 20  # 20 words
        chunks = processor.create_chunks(text, "test.txt")

        assert len(chunks) >= 2

        # Check overlap between first two chunks
        if len(chunks) >= 2:
            chunk1_words = chunks[0]["content"].split()
            chunk2_words = chunks[1]["content"].split()

            # Last 3 words of chunk1 should be first 3 words of chunk2
            overlap_words1 = chunk1_words[-3:]
            overlap_words2 = chunk2_words[:3]
            assert overlap_words1 == overlap_words2

    def test_create_chunks_empty_text(self, document_processor):
        """Test chunking with empty text."""
        chunks = document_processor.create_chunks("", "empty.txt")
        assert chunks == []

    def test_create_chunks_whitespace_only(self, document_processor):
        """Test chunking with whitespace-only text."""
        chunks = document_processor.create_chunks("   \n\t   ", "whitespace.txt")
        assert chunks == []

    def test_create_chunks_single_word(self, document_processor):
        """Test chunking with single word."""
        chunks = document_processor.create_chunks("word", "single.txt")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "word"
        assert chunks[0]["metadata"]["word_count"] == 1

    def test_process_file_not_found(self, document_processor):
        """Test processing non-existent file."""
        with pytest.raises(FileNotFoundError):
            document_processor.process_file(Path("nonexistent.txt"))

    def test_process_file_unsupported_format(self, document_processor):
        """Test processing unsupported file format."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="Unsupported file format"):
                document_processor.process_file(Path("test.xyz"))

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_process_file_txt(self, mock_exists, mock_read_text, document_processor):
        """Test processing .txt file."""
        mock_exists.return_value = True
        mock_read_text.return_value = "This is test content."

        chunks = document_processor.process_file(Path("test.txt"))

        assert len(chunks) == 1
        assert chunks[0]["content"] == "This is test content."
        assert chunks[0]["metadata"]["source"] == "test.txt"

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_process_file_md(self, mock_exists, mock_read_text, document_processor):
        """Test processing .md file."""
        mock_exists.return_value = True
        mock_read_text.return_value = "# Header\nThis is markdown content."

        chunks = document_processor.process_file(Path("test.md"))

        assert len(chunks) == 1
        assert "Header" in chunks[0]["content"]
        assert "markdown content" in chunks[0]["content"]

    @patch("rag_system.core.document_processor.PyPDF2")
    @patch("pathlib.Path.exists")
    def test_parse_pdf_success(self, mock_exists, mock_pypdf2, document_processor):
        """Test successful PDF parsing."""
        mock_exists.return_value = True

        # Mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF page content"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader

        with patch("builtins.open", mock_open()):
            text = document_processor.parse_pdf(Path("test.pdf"))

        assert text == "PDF page content"

    @patch("rag_system.core.document_processor.PyPDF2")
    def test_parse_pdf_import_error(self, mock_pypdf2, document_processor):
        """Test PDF parsing with missing PyPDF2."""
        mock_pypdf2.side_effect = ImportError()

        with pytest.raises(ImportError, match="PyPDF2 is required"):
            document_processor.parse_pdf(Path("test.pdf"))

    @patch("rag_system.core.document_processor.docx")
    @patch("pathlib.Path.exists")
    def test_parse_docx_success(self, mock_exists, mock_docx, document_processor):
        """Test successful DOCX parsing."""
        mock_exists.return_value = True

        # Mock document
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "DOCX paragraph content"
        mock_doc.paragraphs = [mock_paragraph]
        mock_docx.Document.return_value = mock_doc

        text = document_processor.parse_docx(Path("test.docx"))

        assert text == "DOCX paragraph content"

    @patch("rag_system.core.document_processor.docx")
    def test_parse_docx_import_error(self, mock_docx, document_processor):
        """Test DOCX parsing with missing python-docx."""
        mock_docx.side_effect = ImportError()

        with pytest.raises(ImportError, match="python-docx is required"):
            document_processor.parse_docx(Path("test.docx"))

    def test_chunk_metadata_consistency(self, document_processor):
        """Test that chunk metadata is consistent and complete."""
        text = "word " * 100  # 100 words
        chunks = document_processor.create_chunks(text, "test.txt")

        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]

            # Check required fields
            assert "source" in metadata
            assert "chunk_id" in metadata
            assert "start_word" in metadata
            assert "end_word" in metadata
            assert "word_count" in metadata

            # Check values
            assert metadata["source"] == "test.txt"
            assert metadata["chunk_id"] == i
            assert metadata["start_word"] >= 0
            assert metadata["end_word"] > metadata["start_word"]
            assert metadata["word_count"] > 0

    def test_chunk_size_limits(self):
        """Test chunking with various chunk sizes."""
        text = "word " * 20  # 20 words

        # Very small chunks
        processor = DocumentProcessor(chunk_size=2, overlap=0)
        chunks = processor.create_chunks(text, "test.txt")
        assert all(len(chunk["content"].split()) <= 2 for chunk in chunks)

        # Large chunks
        processor = DocumentProcessor(chunk_size=50, overlap=0)
        chunks = processor.create_chunks(text, "test.txt")
        assert len(chunks) == 1  # Should fit in one chunk

    def test_error_handling_during_processing(self, document_processor):
        """Test error handling during file processing."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", side_effect=Exception("Read error")):
                with pytest.raises(Exception, match="Read error"):
                    document_processor.process_file(Path("error.txt"))

    @pytest.mark.parametrize(
        "chunk_size,overlap,expected_chunks",
        [
            (10, 2, 2),  # 20 words, 10 per chunk, 2 overlap
            (5, 1, 4),  # 20 words, 5 per chunk, 1 overlap
            (25, 0, 1),  # 20 words, 25 per chunk (larger than text)
        ],
    )
    def test_chunking_parameters(self, chunk_size, overlap, expected_chunks):
        """Test chunking with different parameters."""
        processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        text = "word " * 20  # 20 words
        chunks = processor.create_chunks(text, "test.txt")

        if chunk_size >= 20:
            # Should fit in one chunk
            assert len(chunks) == 1
        else:
            # Should create multiple chunks
            assert len(chunks) >= expected_chunks
