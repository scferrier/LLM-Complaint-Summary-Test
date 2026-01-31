"""
California Southern District (CASD) specific PDF cleaners.

CASD documents are similar to CAND with margin numbers and "Page X of Y" patterns.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class CASDOrderCleaner(BaseOrderCleaner):
    """Cleaner for California Southern District court order PDFs.

    Handles:
    - Margin line numbers 1-28
    - "Page X of Y" patterns
    - Some garbled sidebar text (less common than CAND)
    """

    def _get_header_patterns(self) -> List[str]:
        """Return CASD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Page\s+\d+\s+of\s+\d+",
            r"Southern District of California",
        ])
        return patterns

    def _get_garbled_text_patterns(self) -> List[str]:
        """Return CASD-specific garbled text patterns."""
        return [
            r"a\s+i\s+n\s+r\s+o\s+f",  # "California" garbled
            r"i\s+l\s+a\s+C",
        ]

    def _remove_garbled_text(self, text: str) -> str:
        """Remove CASD garbled text patterns."""
        for pattern in self._get_garbled_text_patterns():
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove embedded line numbers
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text


class CASDComplaintCleaner(BaseComplaintCleaner):
    """Cleaner for California Southern District complaint PDFs."""

    def _get_header_patterns(self) -> List[str]:
        """Return CASD-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Southern District of California",
            r"Page\s+\d+\s+of\s+\d+",
        ])
        return patterns
