"""
PDF text extraction and cleaning utility for legal documents.

This utility extracts text from PDF files and cleans it for LLM processing
and embedding creation. It handles headers, footers, page numbers,
case captions, signature blocks, and other legal document artifacts.
"""

import os
import re
import pdfplumber
from typing import List, Tuple, Optional
import textwrap
from pathlib import Path


class PDFTextProcessor:
    """Process legal PDF documents for LLM consumption."""

    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 200):
        """
        Initialize the PDF processor.

        Args:
            max_chunk_size: Maximum characters per chunk for large documents
            overlap_size: Character overlap between chunks to maintain context
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        # Patterns for cleaning - case headers and page identifiers
        self.header_patterns = [
            # Case headers with various formats
            r"Case\s+\d+:\d{2}-cv-\d+-\S+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s*ID[.\d]+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+\d+:\d{2}-cv-\d+-\S+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+\d+:\d{2}-cv-\d+\s+Document\s+\d+\s+Filed\s+\d+/\d+/\d+\s+Page\s+\d+\s+of\s+\d+",
            r"Case\s+No\.\s*\d+:\d{2}-cv-\d+-\S+",  # Case No. format
            r"Case\s+No\.\s*\d+-cv-\d+",  # Shorter case number format
            r"\w+\s+v\.\s+\w+.*Doc\.\s*\d+",
            r"\d+\s+Civ\.\s+\d+\s*\(\w{3}\)",
            r"Dockets\.Justia\.com",
            r"^\s*\d+-\d+-\d+\.v\d+\s*$",  # Document control numbers like 4865-0944-1967.v1
            r"^-\s*\d+\s*$",  # Page number dashes
            # Document title lines with case numbers (page footers)
            r"^.*:\s*Case\s+No\.\s*\d+:\d{2}-cv-\d+",
            r"^.*COMPLAINT.*:\s*Case\s+No\.",
            r"^.*COMPLAINT.*Case\s+\d+:",
            # Page footers
            r"^\s*-\s*\d+\s*-\s*$",
            r"^.*Page\s*ID[.\d]+\s*Page\s+\d+",
        ]

        self.page_number_patterns = [
            r"Page \d+ of \d+$",
            r"^\d+$",
            r"^-\s*\d+\s*-$",
            r"^[ivxlc]+$",  # Roman numeral page numbers (i, ii, iii, iv, etc.)
            r"^[ivxlc]+\s+of\s+\d+$",  # "i of 81" format
            r"^of\s+\d+$",  # "of 81" fragment (from partial page numbers)
        ]

        # More conservative signature block patterns - only match clear signature indicators
        self.signature_block_patterns = [
            r"Respectfully\s+submitted",
            r"^/s/\s+",  # Electronic signature
            r"^\s*_+\s*$",  # Signature underline
            r"United States District Judge",
            r"SO ORDERED\.$",
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Raw extracted text
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
        Clean legal document text by removing headers, footers, page numbers,
        case captions, signature blocks, and other artifacts.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
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

            # Clean line content
            cleaned_line = self._clean_line(stripped_line)

            # Skip if line becomes empty after cleaning
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        # Join lines and post-process
        full_text = "\n".join(cleaned_lines)
        full_text = self._post_process_text(full_text)

        return full_text

    def _is_header(self, line: str) -> bool:
        """Check if line is a header."""
        for pattern in self.header_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    def _is_page_number(self, line: str) -> bool:
        """Check if line is a page number."""
        for pattern in self.page_number_patterns:
            if re.fullmatch(pattern, line.strip()):
                return True
        return False

    def _clean_line(self, line: str) -> str:
        """Clean individual line content."""
        # Remove line numbers in gutters (patterns like "1.", "2.", "1 ", "25 " etc. at start)
        line = re.sub(r"^\d+\.\s+", "", line)  # "1. " format
        line = re.sub(r"^\d+\s+", "", line)     # "1 " or "25 " format (margin line numbers)

        # Remove gutter numbers in brackets
        line = re.sub(r"\[\d+\]", "", line)

        # Fix common OCR errors
        line = line.replace("fi", "fi")
        line = line.replace("fl", "fl")
        line = line.replace("ﬂ", "fl")
        line = line.replace("ﬁ", "fi")

        # Remove excessive whitespace
        line = re.sub(r"\s+", " ", line)

        return line.strip()

    def _post_process_text(self, text: str) -> str:
        """Post-process the full text."""
        # Remove signature blocks at the end
        text = self._remove_signature_blocks(text)

        # Fix paragraph spacing
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove case captions (usually at beginning)
        text = self._remove_case_caption(text)

        # Remove table of contents
        text = self._remove_table_of_contents(text)

        # Remove table of authorities
        text = self._remove_table_of_authorities(text)

        # Remove any remaining TOC-style dot leader lines
        text = self._remove_toc_dot_leaders(text)

        # Join lines into continuous prose (remove newlines, normalize whitespace)
        text = self._join_to_prose(text)

        # Final cleanup - remove any remaining dot leaders that survived joining
        text = self._remove_dot_leaders_from_prose(text)

        return text.strip()

    def _remove_case_caption(self, text: str) -> str:
        """Remove case caption from beginning of text."""
        lines = text.split("\n")
        caption_end = 0

        # Look for start of substantive content
        for i, line in enumerate(lines[:100]):  # Check first 100 lines
            line = line.strip()

            # Look for table of contents or first substantive section
            if (
                re.search(r"^TABLE OF CONTENTS$", line, re.IGNORECASE)
                or re.search(r"^I\.\s+INTRODUCTION", line, re.IGNORECASE)
                or re.search(r"^INTRODUCTION$", line, re.IGNORECASE)
                or re.search(r"^COUNT\s+[IVX]+", line, re.IGNORECASE)
                or re.search(r"^BACKGROUND$", line, re.IGNORECASE)
                or re.search(r"^FACTS$", line, re.IGNORECASE)
            ):
                caption_end = i
                break

            # Skip if this looks like part of case caption
            if (
                re.search(r"UNITED STATES DISTRICT COURT", line, re.IGNORECASE)
                or re.search(r"Civil Action No\.", line, re.IGNORECASE)
                or re.search(r" Plaintiff\s*$", line, re.IGNORECASE)
                or re.search(r" Defendants?\s*$", line, re.IGNORECASE)
                or re.search(r"^v\.?\s+|^vs\.?\s+", line, re.IGNORECASE)
                or re.search(r"^\s*x\s*$", line)
                or re.search(r"CLASS ACTION", line, re.IGNORECASE)
                or re.search(r"COMPLAINT", line, re.IGNORECASE)
                or re.search(r"AMENDED COMPLAINT", line, re.IGNORECASE)
                or re.search(r"VIOLATIONS OF THE", line, re.IGNORECASE)
                or re.search(r"SECURITIES LAWS", line, re.IGNORECASE)
                or re.search(r"DEMAND FOR JURY TRIAL", line, re.IGNORECASE)
                or re.search(r"^\s*[A-Z0-9-]+\.v\d+\s*$", line)
            ):  # Document control numbers
                continue

        # If we didn't find a clear start, skip the first 20 lines as they're usually caption
        if caption_end == 0:
            caption_end = 20

        if caption_end > 0:
            return "\n".join(lines[caption_end:])
        return text

    def _remove_signature_blocks(self, text: str) -> str:
        """Remove signature blocks at the end of documents."""
        lines = text.split("\n")
        signature_start = len(lines)

        # Only look at the very last 30 lines for signature blocks
        search_start = max(0, len(lines) - 30)

        for i in range(search_start, len(lines)):
            line = lines[i]
            if any(
                re.search(pattern, line, re.IGNORECASE)
                for pattern in self.signature_block_patterns
            ):
                signature_start = i
                break

        if signature_start < len(lines):
            return "\n".join(lines[:signature_start])
        return text

    def _remove_table_of_contents(self, text: str) -> str:
        """Remove table of contents section from text."""
        lines = text.split("\n")
        toc_start = None
        toc_end = None

        for i, line in enumerate(lines):
            stripped = line.strip().upper()

            # Detect TOC start
            if toc_start is None and re.search(
                r"^TABLE\s+OF\s+CONTENTS?$", stripped
            ):
                toc_start = i
                continue

            # If we're in TOC, look for end markers
            if toc_start is not None and toc_end is None:
                # TOC ends when we hit substantive content that starts with a number
                # (like "1. This is a federal securities class action...")
                if re.search(r"^\d+\.\s+\w+.{50,}", line.strip()):
                    toc_end = i
                    break
                # Or when we hit a long substantive paragraph
                if (
                    len(line.strip()) > 100
                    and not re.search(r"\.\s*\d+\s*$", line)
                    and not re.search(r"\.{2,}\s*\d+\s*$", line)
                    and not re.search(r"\.{5,}", line)
                ):
                    toc_end = i
                    break

        # Remove TOC section if found
        if toc_start is not None:
            if toc_end is None:
                # If no clear end, assume TOC is max 100 lines
                toc_end = min(toc_start + 100, len(lines))
            return "\n".join(lines[:toc_start] + lines[toc_end:])
        return text

    def _remove_table_of_authorities(self, text: str) -> str:
        """Remove table of authorities section from text."""
        lines = text.split("\n")
        toa_start = None
        toa_end = None

        for i, line in enumerate(lines):
            stripped = line.strip().upper()

            # Detect TOA start
            if toa_start is None and re.search(
                r"^TABLE\s+OF\s+AUTHORITIES$", stripped
            ):
                toa_start = i
                continue

            # If we're in TOA, look for end markers
            if toa_start is not None and toa_end is None:
                # TOA ends when we hit substantive content headers
                if re.search(
                    r"^(I\.\s+)?INTRODUCTION$|^SUMMARY$|^BACKGROUND$|"
                    r"^FACTUAL\s+(BACKGROUND|ALLEGATIONS)$|^PARTIES$|"
                    r"^NATURE\s+OF\s+(THE\s+)?ACTION$|^PRELIMINARY\s+STATEMENT$|"
                    r"^COUNT\s+[IVX]+|^JURISDICTION|^TABLE\s+OF\s+CONTENTS?$",
                    stripped,
                ):
                    toa_end = i
                    break
                # Also end if we see a line that looks like substantive paragraph
                if (
                    len(line.strip()) > 100
                    and not re.search(r"\.\s*\d+\s*$", line)
                    and not re.search(r"\.{2,}\s*\d+\s*$", line)
                    and not re.search(r"passim|supra|infra", line, re.IGNORECASE)
                ):
                    toa_end = i
                    break

        # Remove TOA section if found
        if toa_start is not None:
            if toa_end is None:
                # If no clear end, assume TOA is max 100 lines
                toa_end = min(toa_start + 100, len(lines))
            return "\n".join(lines[:toa_start] + lines[toa_end:])
        return text

    def _join_to_prose(self, text: str) -> str:
        """Join text into continuous prose by removing newlines and extra whitespace."""
        # Replace newlines with spaces
        text = re.sub(r"\n+", " ", text)
        # Normalize all whitespace to single spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _remove_toc_dot_leaders(self, text: str) -> str:
        """Remove table of contents style lines with dot leaders (before joining to prose)."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip lines that look like TOC entries with dot leaders
            # Pattern: text followed by dots/periods and a page number
            if re.search(r"\.{3,}\s*\d+\s*$", line):  # ... followed by number
                continue
            if re.search(r"(\.\s+){3,}\d+\s*$", line):  # . . . followed by number
                continue
            if re.search(r"\.{2,}\s*\d+\s*$", line):  # .. followed by number
                continue
            # Catch TOC lines with many dots anywhere in the line
            if re.search(r"\.{5,}", line):  # Line with 5+ consecutive dots
                continue
            # Page number patterns like "Page(s)" at start indicating TOC
            if re.search(r"^\s*Page\(s\)\s*$", line, re.IGNORECASE):
                continue
            # Skip short lines that are just "of XX" (page number fragments)
            if re.search(r"^\s*of\s+\d+\s*$", stripped, re.IGNORECASE):
                continue
            # Skip lines that look like TOC section letters (A., B., C., etc.) without content
            if re.search(r"^[A-Z]\.\s*$", stripped):
                continue
            # Skip lines that are just numbers (subsection numbers from TOC)
            if re.search(r"^\d+\.\s*$", stripped):
                continue
            # Skip TOC-style section headings: lines starting with letter (A., B.)
            # that are short and end abruptly, but NOT numbered paragraphs
            if (
                re.search(r"^[A-Z]\.\s+\w+", stripped)
                and len(stripped) < 80
                and not re.search(r"[.!?]\s*$", stripped)
            ):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _remove_dot_leaders_from_prose(self, text: str) -> str:
        """Remove any remaining TOC-style dot leader patterns from continuous prose."""
        # Remove patterns like "Section Name . . . . . 25" or "Introduction...........12"
        # These patterns: text + multiple dots/spaces + number
        text = re.sub(r"(\s*\.){4,}\s*\d+", "", text)  # . . . . 25
        text = re.sub(r"\.{4,}\s*\d+", "", text)  # .....25
        text = re.sub(r"(\s*\.){3,}\s*\d+", "", text)  # . . . 25
        text = re.sub(r"\.{3,}\s*\d+", "", text)  # ...25

        # Clean up any resulting double spaces
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

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

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # If this single paragraph is too long, split it
            if len(paragraph) > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split the long paragraph by sentences or words
                self._split_long_paragraph(paragraph, chunks)
            # If adding this paragraph would exceed chunk size
            elif len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap if possible
                if chunks:
                    # Get overlap from previous chunk
                    words = chunks[-1].split()
                    overlap_words = []
                    word_count = 0
                    for word in reversed(words):
                        if word_count + len(word) + 1 > self.overlap_size:
                            break
                        overlap_words.insert(0, word)
                        word_count += len(word) + 1

                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_paragraph(self, paragraph: str, chunks: List[str]):
        """Split a very long paragraph into smaller chunks."""
        start = 0
        while start < len(paragraph):
            # Create chunk
            end = min(start + self.max_chunk_size, len(paragraph))

            # Try to break at a sentence or word boundary
            if end < len(paragraph):
                # Look for sentence boundary
                sentence_end = paragraph.rfind(".", start, end)
                if sentence_end > start + self.max_chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for space
                    space_pos = paragraph.rfind(" ", start, end)
                    if space_pos > start + self.max_chunk_size // 2:
                        end = space_pos

            chunk = paragraph[start:end].strip()
            if chunk:
                # Add overlap if we have previous chunks
                if chunks and self.overlap_size > 0:
                    last_chunk = chunks[-1]
                    if len(last_chunk) > self.overlap_size:
                        overlap_text = last_chunk[-self.overlap_size :]
                        # Find word boundary for overlap
                        space_pos = overlap_text.find(" ")
                        if space_pos > 0:
                            overlap_text = overlap_text[space_pos + 1 :]
                        chunk = overlap_text + " " + chunk

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

        # Extract raw text
        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(raw_text)} characters")

        # Clean text
        cleaned_text = self.clean_text(raw_text)
        print(f"Cleaned to {len(cleaned_text)} characters")

        # Split into chunks if needed
        chunks = self.split_into_chunks(cleaned_text)
        print(f"Split into {len(chunks)} chunks")

        # Save to files if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(pdf_path).stem

            # Save single cleaned text file
            full_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # Optionally save chunks
            if save_chunks:
                for i, chunk in enumerate(chunks):
                    chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.txt")
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        f.write(chunk)

            print(f"Saved to {output_dir}")

        return cleaned_text, chunks


def main():
    """Example usage of the PDF processor."""
    processor = PDFTextProcessor()

    # Process all PDFs in the Selected Cases directory
    base_dir = "Selected Cases"
    output_dir = "Processed_Text"

    for category in ["Compliants", "Orders/PDFs"]:
        input_dir = os.path.join(base_dir, category)
        category_output = os.path.join(output_dir, category.replace("/", "_"))

        if os.path.exists(input_dir):
            print(f"\nProcessing {category}...")
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                try:
                    processor.process_pdf(pdf_path, category_output)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")


if __name__ == "__main__":
    main()
