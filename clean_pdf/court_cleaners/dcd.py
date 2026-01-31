"""
District of Columbia (DCD) specific PDF cleaners.

DCD documents use "Dkt. #" and "Civil Action No." patterns.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class DCDOrderCleaner(BaseOrderCleaner):
    """Cleaner for District of Columbia court order PDFs.

    Handles:
    - Docket notation: "Dkt. #"
    - "Civil Action No." format
    """

    def _get_header_patterns(self) -> List[str]:
        """Return DCD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Dkt\.?\s*#?\s*\d+",
            r"Civil Action No\.\s*\d+[-:]\d+",
            r"DISTRICT OF COLUMBIA",
        ])
        return patterns

    def _get_garbled_text_patterns(self) -> List[str]:
        """Return DCD-specific garbled text patterns."""
        return []

    def _remove_garbled_text(self, text: str) -> str:
        """DCD typically has clean OCR."""
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        return text


class DCDComplaintCleaner(BaseComplaintCleaner):
    """Cleaner for District of Columbia complaint PDFs."""

    def _get_header_patterns(self) -> List[str]:
        """Return DCD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Dkt\.?\s*#?\s*\d+",
            r"Civil Action No\.\s*\d+[-:]\d+",
            r"DISTRICT OF COLUMBIA",
        ])
        return patterns
