"""
Base order cleaner class for court order documents.

This module provides the base class for cleaning court orders,
extending BasePDFCleaner with order-specific functionality.
"""

import re
from abc import abstractmethod
from typing import List

from .base import BasePDFCleaner


class BaseOrderCleaner(BasePDFCleaner):
    """Base class for court order PDF cleaning.

    Extends BasePDFCleaner with order-specific functionality including:
    - Case caption removal
    - Signature block removal
    - Section structure preservation (I. BACKGROUND, II. LEGAL STANDARD, etc.)
    - Garbled sidebar text handling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Common order signature patterns
        self._signature_patterns = [
            r"Dated\s+this\s+\d+.*day\s+of\s+\w+,\s+\d{4}",
            r"United States District Judge",
            r"United States Magistrate Judge",
            r"SO ORDERED\.$",
            r"IT IS SO ORDERED\.$",
            r"^\s*/s/\s*",  # Electronic signature
            r"Honorable\s+\w+",
        ]

        # Pattern for section headers (preserved during prose joining)
        self._section_header_pattern = re.compile(
            r'^(I{1,3}V?|V?I{0,3}|[A-Z])\.\s+[A-Z]|'
            r'^(BACKGROUND|INTRODUCTION|DISCUSSION|CONCLUSION|LEGAL STANDARD|'
            r'ANALYSIS|FACTS|PARTIES|ORDER|PROCEDURAL HISTORY)$',
            re.IGNORECASE
        )

    def _get_header_patterns(self) -> List[str]:
        """Return order-specific header patterns."""
        return [
            r"^\d+\s+WO\s*$",  # "1 WO" at top of orders
        ]

    def _post_process(self, text: str) -> str:
        """Post-process order text."""
        # Remove case caption at beginning
        text = self._remove_case_caption(text)

        # Remove signature block at end
        text = self._remove_signature_block(text)

        # Fix paragraph spacing
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Join into prose while preserving section structure
        text = self._join_to_prose_preserve_sections(text)

        # Remove garbled text (subclass-specific)
        text = self._remove_garbled_text(text)

        return text.strip()

    def _remove_case_caption(self, text: str) -> str:
        """Remove case caption from beginning of order."""
        lines = text.split("\n")
        caption_end = 0

        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            line_stripped = line.strip()

            # Look for start of substantive content - section headers
            # Handles: I. BACKGROUND, I. FACTUAL BACKGROUND, I FACTUAL BACKGROUND, etc.
            if re.search(r"^I\.?\s+(FACTUAL\s+)?(BACKGROUND|INTRODUCTION|PROCEDURAL)", line_stripped, re.IGNORECASE):
                caption_end = i
                break
            # Standalone headers (with optional footnote number)
            if re.search(r"^(BACKGROUND|INTRODUCTION|PROCEDURAL\s+HISTORY|FACTUAL\s+BACKGROUND|FACTS)\d*$", line_stripped, re.IGNORECASE):
                caption_end = i
                break
            # Numbered sections: "1. Background"
            if re.search(r"^\d+\.\s*(BACKGROUND|FACTS)", line_stripped, re.IGNORECASE):
                caption_end = i
                break
            # Check for first substantive paragraph (but NOT footnotes which start with numbers or "The following facts")
            if (re.search(r"^(Two|Three|This|Plaintiff|Defendant|Before|In\s+\d{4})", line_stripped)
                and len(line_stripped) > 80
                and not re.search(r"^The\s+following\s+facts", line_stripped, re.IGNORECASE)):
                caption_end = i
                break

        if caption_end > 0:
            return "\n".join(lines[caption_end:])
        return text

    def _remove_signature_block(self, text: str) -> str:
        """Remove judge signature block at end of order."""
        lines = text.split("\n")
        sig_start = len(lines)

        # Look in last 20 lines for signature markers
        search_start = max(0, len(lines) - 20)
        for i in range(search_start, len(lines)):
            line = lines[i]
            for pattern in self._signature_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sig_start = i
                    break
            if sig_start < len(lines):
                break

        if sig_start < len(lines):
            return "\n".join(lines[:sig_start])
        return text

    def _join_to_prose_preserve_sections(self, text: str) -> str:
        """Join text into prose but preserve section structure markers."""
        lines = text.split("\n")
        result_parts = []
        current_paragraph = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a section header
            if self._section_header_pattern.match(stripped):
                # Save current paragraph
                if current_paragraph:
                    result_parts.append(" ".join(current_paragraph))
                    current_paragraph = []
                # Add section marker with spacing
                result_parts.append(f"\n\n{stripped}\n")
            else:
                current_paragraph.append(stripped)

        # Add final paragraph
        if current_paragraph:
            result_parts.append(" ".join(current_paragraph))

        text = " ".join(result_parts)
        # Clean up spacing
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)

        return text.strip()

    def _remove_garbled_text(self, text: str) -> str:
        """Remove garbled text from the document.

        Default implementation - subclasses override for court-specific patterns.
        """
        # Remove isolated embedded line numbers (1-28 followed by capital letter)
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        return text

    @abstractmethod
    def _get_garbled_text_patterns(self) -> List[str]:
        """Return court-specific garbled text patterns.

        Subclasses must implement this for court-specific OCR artifacts.
        """
        pass

    def _is_garbled_line(self, line: str) -> bool:
        """Check if line is purely garbled text (no real content).

        Override in subclass for court-specific detection.
        """
        return False

    def _should_skip_line(self, line: str) -> bool:
        """Skip garbled lines."""
        return self._is_garbled_line(line)
