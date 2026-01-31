"""
California Northern District (CAND) specific PDF cleaners.

CAND documents often have garbled sidebar text from vertical "Northern District
of California" text that OCR incorrectly extracts as spaced letters.
"""

import re
from typing import List

from ..order_base import BaseOrderCleaner
from ..complaint_base import BaseComplaintCleaner


class CANDOrderCleaner(BaseOrderCleaner):
    """Cleaner for California Northern District court order PDFs.

    Handles:
    - Garbled sidebar patterns: "u o r o", "a t r i n", "t a t s i D"
    - Left margin line numbers 1-28
    - Inline garbled fragments: "-17 ", "ee", "th 14"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # CAND-specific garbled sidebar patterns
        self._garbled_sidebar_patterns = [
            r"u\s+o\s+r\s+o",
            r"a\s+t\s+r\s+i\s+n",
            r"t\s+a\s+t\s+s\s+i\s+D",
            r"r\s+t\s+s\s+f\s+o",
            r"Cf\s+i\s+t\s+l\s+a",
            r"i\s+r\s+t\s+s",
            r"si\s+r\s+et",
            r"th\s+i\s+nt\s+r",
            r"S\s+d\s+n\s+r\s+ee",
            r"cC\s+i\s+l",
        ]

        # Pattern for purely garbled lines
        self._garbled_line_pattern = re.compile(
            r'^(?:[a-zA-Z]\s+){2,}[a-zA-Z]?\s*\d*\s*$'
        )

    def _get_garbled_text_patterns(self) -> List[str]:
        """Return CAND-specific garbled text patterns."""
        return self._garbled_sidebar_patterns

    def _pre_process(self, text: str) -> str:
        """Pre-process to remove multi-line garbled sidebar blocks."""
        return self._remove_garbled_blocks(text)

    def _remove_garbled_blocks(self, text: str) -> str:
        """Remove garbled sidebar text using line-by-line filtering."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            # Skip very short lines that are just letters/numbers (garbled fragments)
            if len(stripped) <= 8:
                if re.match(r'^[a-zA-Z]{1,2}$', stripped):
                    continue
                if re.match(r'^[a-zA-Z]{1,2}\s*\d{1,2}$', stripped):
                    continue
                if re.match(r'^[a-zA-Z]\s+[a-zA-Z](\s+[a-zA-Z])*\s*\d*$', stripped):
                    continue

            # Skip lines that are garbled "Northern District of California" sidebar
            garbled_sidebar = [
                r'^a\s+t\s+r\s+i\s+n$',
                r'^u\s+o\s+r\s+o\s*\d*$',
                r'^i\s+r\s+t\s+s\s+f\s+o$',
                r'^i\s+D\s+t\s+c\s*\d*$',
                r'^t\s+a\s+t\s+s\s+i\s+D\s*\d*$',
                r'^r\s+e[e\s]*\d*$',
                r'^t\s*h\s*i\s*n\s*t$',
                r'^Uo\s*N\s*\d*$',
                r'^Cf\s*i?\s*t?\s*l?\s*a?$',
                r'^cC\s*\d*$',
                r'^si\s*r?\s*et?$',
                r'^S\s*d\s*e?\s*n?$',
            ]

            is_garbled = False
            for pattern in garbled_sidebar:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_garbled = True
                    break

            if is_garbled:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _is_garbled_line(self, line: str) -> bool:
        """Check if line is purely garbled sidebar text."""
        if self._garbled_line_pattern.match(line):
            return True
        if len(line) < 15 and re.match(r'^[a-zA-Z\s]+\d*$', line):
            words = line.split()
            single_letters = sum(1 for w in words if len(w) == 1 or w.isdigit())
            if single_letters >= len(words) * 0.5:
                return True
        return False

    def _remove_garbled_text(self, text: str) -> str:
        """Remove garbled vertical sidebar text from the document."""
        # Remove specific garbled patterns
        for pattern in self._garbled_sidebar_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove shorter garbled fragments that appear in context
        text = re.sub(r'-\d{1,2}\s+', '-', text)  # "publicly-17 traded" -> "publicly-traded"
        text = re.sub(r'-\s+ee\s+', '-', text)    # "publicly- ee traded" -> "publicly-traded"
        text = re.sub(r'\s+th\s+to\b', ' to', text)  # "seeks th to" -> "seeks to"

        # Remove inline garbled sidebar fragments
        text = re.sub(r'\s+cC\s+\d{1,2}\s+', ' ', text)
        text = re.sub(r'\s+t\s+r\s+i\s+n\s+', ' ', text)
        text = re.sub(r'\s+u\s+o\s+C?\s*r\s+o\s+f?\s*i?\s+', ' ', text)
        text = re.sub(r'\s+e\s+t\s+a\s+t\s+t?\s*s\s+i\s+D\s+\d+\s+i\s+\d+\s+g\s+n\s+e[^A-Z]*', ' ', text)
        text = re.sub(r'\s+(?:[a-zA-Z]\s+){4,}[a-zA-Z]?\s*\d*\s+', ' ', text)

        # Various other garbled fragments
        text = re.sub(r'\s+ee\s+th\s+', ' ', text)
        text = re.sub(r'\s+Cf\s+cC\s+', ' ', text)
        text = re.sub(r'\s+D\s+t\s+c\s+', ' ', text)
        text = re.sub(r'\s+Uo\s+N\s+', ' ', text)
        text = re.sub(r'\s+si\s+et\s+', ' ', text)

        # Remove isolated single/double letter fragments between sentences
        text = re.sub(r'([.!?])\s+[A-Za-z]{1,2}\s+([A-Z])', r'\1 \2', text)

        # Remove isolated line numbers after garbled text removal
        text = re.sub(r'\s+(\d{1,2})\s+([a-z][a-z])', r' \2', text)
        text = re.sub(r'\s+(\d{1,2})\s+([A-Z][a-z])', r' \2', text)

        # Clean up multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)

        return text

    def _clean_line(self, line: str) -> str:
        """Clean individual line with CAND-specific handling."""
        # Remove garbled sidebar prefixes
        garbled_prefix = re.match(
            r'^(?:[a-zA-Z]\s+){2,}[a-zA-Z]?\s*(\d{1,2})?\s+([A-Z][a-z].+)$',
            line
        )
        if garbled_prefix:
            line = garbled_prefix.group(2)

        # Call parent implementation for standard cleaning
        return super()._clean_line(line)


class CANDComplaintCleaner(BaseComplaintCleaner):
    """Cleaner for California Northern District complaint PDFs."""

    def _get_header_patterns(self) -> List[str]:
        """Return CAND-specific header patterns."""
        patterns = super()._get_header_patterns()
        patterns.extend([
            r"Northern District of California",
        ])
        return patterns
