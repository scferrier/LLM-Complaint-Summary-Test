"""
Base PDF cleaner class with shared functionality for legal documents.

This module provides the abstract base class for all PDF cleaners,
implementing common text extraction and cleaning methods.
"""

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber


class BasePDFCleaner(ABC):
    """Abstract base class for PDF text extraction and cleaning.

    Provides shared functionality for extracting and cleaning text from
    legal PDF documents. Subclasses implement document-type-specific
    cleaning logic.
    """

    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 200):
        """
        Initialize the PDF cleaner.

        Args:
            max_chunk_size: Maximum characters per chunk for large documents
            overlap_size: Character overlap between chunks to maintain context
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        # Common header patterns across all legal documents
        self._common_header_patterns = [
            r"Case\s+\d+:\d{2}-cv-\d+-\S+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s*ID[.\d]+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+\d+:\d{2}-cv-\d+-\S+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+\d+:\d{2}-cv-\d+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+No\.\s*\d+:\d{2}-cv-\d+-\S+",
            r"Case\s+No\.\s*\d+-cv-\d+",
            r"Dockets\.Justia\.com",
        ]

        # Common page number patterns
        self._common_page_number_patterns = [
            r"^Page \d+ of \d+$",
            r"^\d+$",
            r"^-\s*\d+\s*-$",
            r"^[ivxlc]+$",  # Roman numerals
            r"^[ivxlc]+\s+of\s+\d+$",
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF file using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Raw extracted text

        Raises:
            Exception: If text extraction fails
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        all_text.append(page_text)
                return "\n\n".join(all_text)
        except Exception as e:
            raise Exception(f"Error extracting text from {pdf_path}: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Clean document text using template method pattern.

        Subclasses can override hook methods to customize behavior.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Pre-process hook (subclass can override)
        text = self._pre_process(text)

        # Line-by-line cleaning
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped_line = line.strip()

            # Skip empty lines
            if not stripped_line:
                continue

            # Remove headers
            if self._is_header(stripped_line):
                continue

            # Remove page numbers
            if self._is_page_number(stripped_line):
                continue

            # Apply subclass-specific line filter
            if self._should_skip_line(stripped_line):
                continue

            # Clean line content
            cleaned_line = self._clean_line(stripped_line)

            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        # Join lines and post-process
        full_text = "\n".join(cleaned_lines)
        full_text = self._post_process(full_text)

        return full_text

    def _pre_process(self, text: str) -> str:
        """Hook for pre-processing before line-by-line cleaning.

        Override in subclass to add pre-processing steps.
        """
        return text

    def _post_process(self, text: str) -> str:
        """Hook for post-processing after line-by-line cleaning.

        Override in subclass to add post-processing steps.
        """
        # Fix paragraph spacing
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _should_skip_line(self, line: str) -> bool:
        """Hook for subclass-specific line filtering.

        Override in subclass to skip additional line types.
        """
        return False

    def _is_header(self, line: str) -> bool:
        """Check if line is a document header."""
        all_patterns = self._common_header_patterns + self._get_header_patterns()
        for pattern in all_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    def _is_page_number(self, line: str) -> bool:
        """Check if line is a page number."""
        all_patterns = self._common_page_number_patterns + self._get_page_number_patterns()
        for pattern in all_patterns:
            if re.fullmatch(pattern, line.strip(), re.IGNORECASE):
                return True
        return False

    def _clean_line(self, line: str) -> str:
        """Clean individual line content.

        Handles margin line numbers, OCR fixes, and whitespace.
        """
        # Remove margin line numbers (1-28 typically in legal docs)
        line = self._remove_margin_numbers(line)

        # Fix common OCR ligature errors
        line = self._fix_ocr_errors(line)

        # Normalize whitespace
        line = re.sub(r"\s+", " ", line)

        return line.strip()

    def _remove_margin_numbers(self, line: str) -> str:
        """Remove margin line numbers (1-28) from start of line."""
        # Pattern: line starts with 1-2 digit number followed by space(s) and content
        match = re.match(r'^(\d{1,2})\s+(.+)$', line)
        if match:
            line_num = int(match.group(1))
            content = match.group(2)
            # Only strip if it looks like a margin number (1-28 range)
            # and content doesn't look like a numbered list (e.g., "1. Item")
            if 1 <= line_num <= 28 and not re.match(r'^[.)]', content):
                return content
        return line

    def _fix_ocr_errors(self, line: str) -> str:
        """Fix common OCR errors in text."""
        # Ligature fixes
        line = line.replace("fi", "fi")
        line = line.replace("fl", "fl")
        line = line.replace("\ufb02", "fl")  # ﬂ
        line = line.replace("\ufb01", "fi")  # ﬁ
        return line

    def _join_to_prose(self, text: str) -> str:
        """Convert text to continuous prose by removing newlines."""
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @abstractmethod
    def _get_header_patterns(self) -> List[str]:
        """Return document-type-specific header patterns.

        Subclasses must implement this to add their specific patterns.
        """
        pass

    def _get_page_number_patterns(self) -> List[str]:
        """Return document-type-specific page number patterns.

        Override in subclass to add specific patterns.
        """
        return []

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split large text into manageable chunks for LLM processing.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                self._split_long_paragraph(paragraph, chunks)
            elif len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = self._get_overlap(chunks) + paragraph if chunks else paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap(self, chunks: List[str]) -> str:
        """Get overlap text from previous chunk."""
        if not chunks:
            return ""
        words = chunks[-1].split()
        overlap_words = []
        word_count = 0
        for word in reversed(words):
            if word_count + len(word) + 1 > self.overlap_size:
                break
            overlap_words.insert(0, word)
            word_count += len(word) + 1
        return " ".join(overlap_words) + "\n\n" if overlap_words else ""

    def _split_long_paragraph(self, paragraph: str, chunks: List[str]):
        """Split a very long paragraph into smaller chunks."""
        start = 0
        while start < len(paragraph):
            end = min(start + self.max_chunk_size, len(paragraph))

            if end < len(paragraph):
                sentence_end = paragraph.rfind(".", start, end)
                if sentence_end > start + self.max_chunk_size // 2:
                    end = sentence_end + 1
                else:
                    space_pos = paragraph.rfind(" ", start, end)
                    if space_pos > start + self.max_chunk_size // 2:
                        end = space_pos

            chunk = paragraph[start:end].strip()
            if chunk:
                if chunks and self.overlap_size > 0:
                    overlap = self._get_overlap(chunks)
                    chunk = overlap + chunk if overlap else chunk
                chunks.append(chunk)

            start = end

    def process_pdf(
        self, pdf_path: str, output_dir: Optional[str] = None, save_chunks: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Process a PDF file: extract, clean, and optionally save results.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save processed text files (optional)
            save_chunks: If True, also save chunk files (default: False)

        Returns:
            Tuple of (cleaned_text, chunks)
        """
        print(f"Processing {pdf_path}...")

        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(raw_text)} characters")

        cleaned_text = self.clean_text(raw_text)
        print(f"  Cleaned to {len(cleaned_text)} characters")

        chunks = self.split_into_chunks(cleaned_text)
        print(f"  Split into {len(chunks)} chunks")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(pdf_path).stem

            full_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            if save_chunks:
                for i, chunk in enumerate(chunks):
                    chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.txt")
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        f.write(chunk)

            print(f"  Saved to {output_dir}")

        return cleaned_text, chunks
