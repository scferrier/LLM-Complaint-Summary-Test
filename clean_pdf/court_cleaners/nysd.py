"""
New York Southern District (NYSD) specific PDF cleaners.

NYSD documents typically have cleaner OCR output but use PageID footer
patterns and Civ. case number format.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class NYSDOrderCleaner(BaseOrderCleaner):
    """Cleaner for New York Southern District court order PDFs.

    Handles:
    - PageID footer patterns: "PageID #:"
    - Civ. case number format
    - Generally cleaner OCR (minimal garbled patterns)
    """

    def _get_header_patterns(self) -> List[str]:
        """Return NYSD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"\d+\s+Civ\.\s+\d+\s*\(\w{3}\)",  # "20 Civ. 4494 (JPC)"
            r"PageID[.\s]*#?:?\s*\d+",
            r"Page\s+\d+\s+of\s+\d+",
        ])
        return patterns

    def _get_garbled_text_patterns(self) -> List[str]:
        """Return NYSD-specific garbled text patterns.

        NYSD documents typically have cleaner OCR, so minimal patterns.
        """
        return []

    def _remove_garbled_text(self, text: str) -> str:
        """NYSD has cleaner OCR, minimal garbled text removal needed."""
        # Just remove embedded line numbers
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        return text


class NYSDComplaintCleaner(BaseComplaintCleaner):
    """Cleaner for New York Southern District complaint PDFs."""

    def _get_header_patterns(self) -> List[str]:
        """Return NYSD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"\d+\s+Civ\.\s+\d+\s*\(\w{3}\)",
            r"PageID[.\s]*#?:?\s*\d+",
            r"SOUTHERN DISTRICT OF NEW YORK",
        ])
        return patterns
