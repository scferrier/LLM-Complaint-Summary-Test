"""
Generic PDF cleaners for unknown or unsupported courts.

Used as fallback when court cannot be detected from filename.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class GenericOrderCleaner(BaseOrderCleaner):
    """Generic cleaner for court order PDFs from unknown courts.

    Uses conservative defaults suitable for most federal district courts.
    Falls back to this for: azd, txnd, flsd, ctd, dde, and others.
    """

    def _get_header_patterns(self) -> List[str]:
        """Return generic header patterns."""
        return [
            r"Page\s+\d+\s+of\s+\d+",
            r"Document\s*#?:?\s*\d+",
        ]

    def _get_garbled_text_patterns(self) -> List[str]:
        """Generic has no court-specific garbled patterns."""
        return []

    def _remove_garbled_text(self, text: str) -> str:
        """Conservative garbled text removal."""
        # Only remove clearly embedded line numbers
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)
        return text


class GenericComplaintCleaner(BaseComplaintCleaner):
    """Generic cleaner for complaint PDFs from unknown courts.

    Uses conservative defaults suitable for most federal district courts.
    """

    def _get_header_patterns(self) -> List[str]:
        """Return generic header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Page\s+\d+\s+of\s+\d+",
            r"Document\s*#?:?\s*\d+",
        ])
        return patterns
