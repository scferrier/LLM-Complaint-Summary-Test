"""
Base complaint cleaner class for legal complaint documents.

This module provides the base class for cleaning legal complaints,
extending BasePDFCleaner with complaint-specific functionality.
"""

import re
from typing import List

from .base import BasePDFCleaner


class BaseComplaintCleaner(BasePDFCleaner):
    """Base class for legal complaint PDF cleaning.

    Extends BasePDFCleaner with complaint-specific functionality including:
    - Table of contents removal
    - Table of authorities removal
    - Case caption removal
    - Signature block removal (attorney signatures)
    - DOT leader removal for TOC-style lines
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Complaint-specific signature patterns
        self._signature_patterns = [
            r"Respectfully\s+submitted",
            r"^/s/\s+",  # Electronic signature
            r"^\s*_+\s*$",  # Signature underline
        ]

    def _get_header_patterns(self) -> List[str]:
        """Return complaint-specific header patterns."""
        return [
            r"\w+\s+v\.\s+\w+.*Doc\.\s*\d+",
            r"\d+\s+Civ\.\s+\d+\s*\(\w{3}\)",
            r"^\s*\d+-\d+-\d+\.v\d+\s*$",  # Document control numbers
            r"^-\s*\d+\s*$",  # Page number dashes
            r"^.*:\s*Case\s+No\.\s*\d+:\d{2}-cv-\d+",
            r"^.*COMPLAINT.*:\s*Case\s+No\.",
            r"^.*COMPLAINT.*Case\s+\d+:",
            r"^\s*-\s*\d+\s*-\s*$",
            r"^.*Page\s*ID[.\d]+\s*Page\s+\d+",
        ]

    def _get_page_number_patterns(self) -> List[str]:
        """Return complaint-specific page number patterns."""
        return [
            r"^of\s+\d+$",  # "of 81" fragment
        ]

    def _post_process(self, text: str) -> str:
        """Post-process complaint text."""
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

        # Join lines into continuous prose
        text = self._join_to_prose(text)

        # Final cleanup - remove any remaining dot leaders
        text = self._remove_dot_leaders_from_prose(text)

        return text.strip()

    def _remove_case_caption(self, text: str) -> str:
        """Remove case caption from beginning of text."""
        lines = text.split("\n")
        caption_end = 0

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
            ):
                continue

            # Check for substantive introductory paragraphs
            if (
                re.search(r"^In\s+\d{4},?\s+", line)
                or re.search(r"^In\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", line, re.IGNORECASE)
                or (re.search(r"^(In|On|The|This|Before|Plaintiff|Defendant|Petitioner|Respondent)", line) and len(line) > 80)
            ):
                caption_end = i
                break

        if caption_end > 0:
            return "\n".join(lines[caption_end:])
        return text

    def _remove_signature_blocks(self, text: str) -> str:
        """Remove signature blocks at the end of documents."""
        lines = text.split("\n")
        signature_start = len(lines)

        # Only look at the very last 30 lines
        search_start = max(0, len(lines) - 30)

        for i in range(search_start, len(lines)):
            line = lines[i]
            if any(
                re.search(pattern, line, re.IGNORECASE)
                for pattern in self._signature_patterns
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
            if toc_start is None and re.search(r"^TABLE\s+OF\s+CONTENTS?$", stripped):
                toc_start = i
                continue

            # If we're in TOC, look for end markers
            if toc_start is not None and toc_end is None:
                # TOC ends when we hit substantive content
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

        if toc_start is not None:
            if toc_end is None:
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
            if toa_start is None and re.search(r"^TABLE\s+OF\s+AUTHORITIES$", stripped):
                toa_start = i
                continue

            # If we're in TOA, look for end markers
            if toa_start is not None and toa_end is None:
                if re.search(
                    r"^(I\.\s+)?INTRODUCTION$|^SUMMARY$|^BACKGROUND$|"
                    r"^FACTUAL\s+(BACKGROUND|ALLEGATIONS)$|^PARTIES$|"
                    r"^NATURE\s+OF\s+(THE\s+)?ACTION$|^PRELIMINARY\s+STATEMENT$|"
                    r"^COUNT\s+[IVX]+|^JURISDICTION|^TABLE\s+OF\s+CONTENTS?$",
                    stripped,
                ):
                    toa_end = i
                    break
                # Also end if we see substantive paragraph
                if (
                    len(line.strip()) > 100
                    and not re.search(r"\.\s*\d+\s*$", line)
                    and not re.search(r"\.{2,}\s*\d+\s*$", line)
                    and not re.search(r"passim|supra|infra", line, re.IGNORECASE)
                ):
                    toa_end = i
                    break

        if toa_start is not None:
            if toa_end is None:
                toa_end = min(toa_start + 100, len(lines))
            return "\n".join(lines[:toa_start] + lines[toa_end:])
        return text

    def _remove_toc_dot_leaders(self, text: str) -> str:
        """Remove table of contents style lines with dot leaders."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip lines that look like TOC entries with dot leaders
            if re.search(r"\.{3,}\s*\d+\s*$", line):
                continue
            if re.search(r"(\.\s+){3,}\d+\s*$", line):
                continue
            if re.search(r"\.{2,}\s*\d+\s*$", line):
                continue
            if re.search(r"\.{5,}", line):
                continue
            if re.search(r"^\s*Page\(s\)\s*$", line, re.IGNORECASE):
                continue
            if re.search(r"^\s*of\s+\d+\s*$", stripped, re.IGNORECASE):
                continue
            if re.search(r"^[A-Z]\.\s*$", stripped):
                continue
            if re.search(r"^\d+\.\s*$", stripped):
                continue
            # Skip TOC-style section headings
            if (
                re.search(r"^[A-Z]\.\s+\w+", stripped)
                and len(stripped) < 80
                and not re.search(r"[.!?]\s*$", stripped)
            ):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _remove_dot_leaders_from_prose(self, text: str) -> str:
        """Remove any remaining TOC-style dot leader patterns from prose."""
        text = re.sub(r"(\s*\.){4,}\s*\d+", "", text)
        text = re.sub(r"\.{4,}\s*\d+", "", text)
        text = re.sub(r"(\s*\.){3,}\s*\d+", "", text)
        text = re.sub(r"\.{3,}\s*\d+", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
