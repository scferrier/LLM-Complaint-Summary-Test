"""
Background section extractor for legal documents.

Extracts the BACKGROUND, FACTS, or similar sections from court orders
for use in summarization tasks.
"""

import re
from typing import Optional, Tuple


class BackgroundExtractor:
    """Extract background/facts sections from legal documents.

    Identifies and extracts the factual background section from court orders,
    which typically appears between the case caption and the legal standard
    or discussion sections.
    """

    def __init__(self):
        # Patterns that indicate the start of a background section
        # These work in both prose (no newlines) and multiline text
        # Handles:
        #   - ALL CAPS or Title Case (BACKGROUND, Background)
        #   - Optional footnote numbers (BACKGROUND1, Background2)
        #   - Roman numerals with/without period (I. BACKGROUND, I BACKGROUND)
        #   - Numbered sections (1. Background)
        self._start_patterns = [
            # Roman numeral with period: "I. BACKGROUND" or "I. Background"
            r"(?:^|\s)I\.\s*(?:BACKGROUND|Background)\d*(?:\s|$)",
            r"(?:^|\s)I\.\s*(?:FACTUAL\s+BACKGROUND|Factual\s+Background)\d*(?:\s|$)",
            r"(?:^|\s)I\.\s*(?:FACTUAL\s+ALLEGATIONS|Factual\s+Allegations)\d*(?:\s|$)",
            r"(?:^|\s)I\.\s*(?:FACTS|Facts)\d*(?:\s|$)",
            r"(?:^|\s)I\.\s*(?:STATEMENT\s+OF\s+FACTS|Statement\s+of\s+Facts)\d*(?:\s|$)",
            # Roman numeral without period: "I BACKGROUND" or "I Factual Background"
            r"(?:^|\s)I\s+(?:BACKGROUND|Background)\d*(?:\s|$)",
            r"(?:^|\s)I\s+(?:FACTUAL\s+BACKGROUND|Factual\s+Background)\d*(?:\s|$)",
            # Standalone headers (ALL CAPS or Title Case with optional footnote)
            r"(?:^|\s)(?:BACKGROUND|Background)\d*(?:\s|$)",
            r"(?:^|\s)(?:FACTUAL\s+BACKGROUND|Factual\s+Background)\d*(?:\s|$)",
            r"(?:^|\s)(?:FACTUAL\s+ALLEGATIONS|Factual\s+Allegations)\d*(?:\s|$)",
            r"(?:^|\s)(?:FACTS|Facts)\d*(?:\s|$)",
            r"(?:^|\s)(?:STATEMENT\s+OF\s+FACTS|Statement\s+of\s+Facts)\d*(?:\s|$)",
            r"(?:^|\s)(?:RELEVANT\s+FACTS|Relevant\s+Facts)\d*(?:\s|$)",
            r"(?:^|\s)(?:PROCEDURAL\s+AND\s+FACTUAL\s+BACKGROUND)\d*(?:\s|$)",
            # Numbered sections: "1. Background"
            r"(?:^|\s)\d+\.\s*(?:BACKGROUND|Background)\d*(?:\s|$)",
        ]

        # Patterns that indicate the end of a background section
        self._end_patterns = [
            r"(?:^|\s)II\.\s+[A-Z]",  # Next Roman numeral section
            r"(?:^|\s)II\s+[A-Z]",    # Roman numeral without period
            r"(?:^|\s)III\.\s+[A-Z]",
            r"(?:^|\s)IV\.\s+[A-Z]",
            r"(?:^|\s)(?:LEGAL\s+STANDARD|Legal\s+Standard)\d*(?:\s|$)",
            r"(?:^|\s)(?:STANDARD\s+OF\s+REVIEW|Standard\s+of\s+Review)\d*(?:\s|$)",
            r"(?:^|\s)(?:DISCUSSION|Discussion)\d*(?:\s|$)",
            r"(?:^|\s)(?:ANALYSIS|Analysis)\d*(?:\s|$)",
            r"(?:^|\s)(?:CONCLUSION|Conclusion)\d*(?:\s|$)",
            r"(?:^|\s)(?:ORDER|Order)\d*(?:\s|$)",
            r"(?:^|\s)(?:RULING|Ruling)\d*(?:\s|$)",
            r"(?:^|\s)(?:APPLICABLE\s+LAW|Applicable\s+Law)\d*(?:\s|$)",
            # Numbered sections for end markers
            r"(?:^|\s)\d+\.\s*(?:LEGAL\s+STANDARD|Legal\s+Standard)",
            r"(?:^|\s)\d+\.\s*(?:DISCUSSION|Discussion)",
            r"(?:^|\s)\d+\.\s*(?:ANALYSIS|Analysis)",
        ]

        # Compile patterns for efficiency
        self._start_pattern = re.compile(
            "|".join(f"({p})" for p in self._start_patterns)
        )
        self._end_pattern = re.compile(
            "|".join(f"({p})" for p in self._end_patterns)
        )

    def extract(self, text: str, include_header: bool = False) -> Optional[str]:
        """Extract the background/facts section from document text.

        Args:
            text: The full document text (cleaned)
            include_header: If True, include the section header (e.g., "I. BACKGROUND")

        Returns:
            The extracted background section text, or None if not found
        """
        boundaries = self.get_boundaries(text)

        if boundaries is None:
            return None

        start_pos, end_pos = boundaries

        # Extract the section
        section = text[start_pos:end_pos].strip()

        # Optionally remove the header
        if not include_header:
            # Remove the header pattern from the start
            # Handle both prose (space-separated) and newline-separated text
            header_match = self._start_pattern.match(section)
            if header_match:
                section = section[header_match.end():].strip()

        return section if section else None

    def get_boundaries(self, text: str) -> Optional[Tuple[int, int]]:
        """Get the start and end positions of the background section.

        Args:
            text: The full document text

        Returns:
            Tuple of (start_position, end_position) or None if not found
        """
        # Find the start of the background section
        start_match = self._start_pattern.search(text)

        if not start_match:
            return None

        start_pos = start_match.start()

        # Find the end of the background section (start of next section)
        # Search from after the start match
        end_match = self._end_pattern.search(text, start_match.end())

        if end_match:
            end_pos = end_match.start()
        else:
            # If no end marker found, take a reasonable chunk
            # (up to 10000 characters or end of text)
            end_pos = min(start_pos + 10000, len(text))

        return (start_pos, end_pos)

    def has_background_section(self, text: str) -> bool:
        """Check if the document contains a background section.

        Args:
            text: The document text

        Returns:
            True if a background section is found
        """
        return self._start_pattern.search(text) is not None

    def get_section_header(self, text: str) -> Optional[str]:
        """Get the header of the background section.

        Args:
            text: The document text

        Returns:
            The section header text (e.g., "I. BACKGROUND") or None
        """
        match = self._start_pattern.search(text)
        if match:
            return match.group(0).strip()
        return None
