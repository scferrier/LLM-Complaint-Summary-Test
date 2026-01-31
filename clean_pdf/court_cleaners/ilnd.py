"""
Illinois Northern District (ILND) specific PDF cleaners.

ILND documents include division info and "Document #:" patterns.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class ILNDOrderCleaner(BaseOrderCleaner):
    """Cleaner for Illinois Northern District court order PDFs.

    Handles:
    - Division info: "EASTERN DIVISION"
    - "Document #:" patterns
    """

    def _get_header_patterns(self) -> List[str]:
        """Return ILND-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"EASTERN\s+DIVISION",
            r"WESTERN\s+DIVISION",
            r"Document\s*#?:?\s*\d+",
            r"Northern District of Illinois",
        ])
        return patterns

    def _get_garbled_text_patterns(self) -> List[str]:
        """Return ILND-specific garbled text patterns."""
        return []

    def _remove_garbled_text(self, text: str) -> str:
        """ILND typically has clean OCR."""
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        return text


class ILNDComplaintCleaner(BaseComplaintCleaner):
    """Cleaner for Illinois Northern District complaint PDFs."""

    def _get_header_patterns(self) -> List[str]:
        """Return ILND-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"EASTERN\s+DIVISION",
            r"WESTERN\s+DIVISION",
            r"Document\s*#?:?\s*\d+",
            r"Northern District of Illinois",
        ])
        return patterns
