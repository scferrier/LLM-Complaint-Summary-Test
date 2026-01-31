"""
Factory functions for creating PDF cleaners.

Provides auto-detection of court from filename and returns the appropriate
cleaner instance.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

from .base import BasePDFCleaner
from .order_base import BaseOrderCleaner
from .complaint_base import BaseComplaintCleaner
from .court_cleaners.cand import CANDOrderCleaner, CANDComplaintCleaner
from .court_cleaners.nysd import NYSDOrderCleaner, NYSDComplaintCleaner
from .court_cleaners.casd import CASDOrderCleaner, CASDComplaintCleaner
from .court_cleaners.dcd import DCDOrderCleaner, DCDComplaintCleaner
from .court_cleaners.ilnd import ILNDOrderCleaner, ILNDComplaintCleaner
from .court_cleaners.generic import GenericOrderCleaner, GenericComplaintCleaner


# Court code to cleaner class mapping
ORDER_CLEANERS: Dict[str, Type[BaseOrderCleaner]] = {
    "cand": CANDOrderCleaner,
    "nysd": NYSDOrderCleaner,
    "casd": CASDOrderCleaner,
    "dcd": DCDOrderCleaner,
    "ilnd": ILNDOrderCleaner,
}

COMPLAINT_CLEANERS: Dict[str, Type[BaseComplaintCleaner]] = {
    "cand": CANDComplaintCleaner,
    "nysd": NYSDComplaintCleaner,
    "casd": CASDComplaintCleaner,
    "dcd": DCDComplaintCleaner,
    "ilnd": ILNDComplaintCleaner,
}

# Patterns for court detection from filename
COURT_PATTERNS = [
    (r"^cand[-_]", "cand"),
    (r"^cand_", "cand"),
    (r"^nysd[-_]", "nysd"),
    (r"^nysd_", "nysd"),
    (r"^casd[-_]", "casd"),
    (r"^casd_", "casd"),
    (r"^dcd[-_]", "dcd"),
    (r"^dcd_", "dcd"),
    (r"^ilnd[-_]", "ilnd"),
    (r"^ilnd_", "ilnd"),
    # Additional courts that use generic cleaners
    (r"^azd[-_]", "azd"),
    (r"^txnd[-_]", "txnd"),
    (r"^flsd[-_]", "flsd"),
    (r"^ctd[-_]", "ctd"),
    (r"^dde[-_]", "dde"),
]


def detect_court(filename: str) -> Optional[str]:
    """Detect court code from filename.

    Args:
        filename: The PDF filename (with or without path)

    Returns:
        Court code (e.g., "cand", "nysd") or None if not detected
    """
    # Get just the filename without path
    basename = os.path.basename(filename).lower()

    for pattern, court_code in COURT_PATTERNS:
        if re.match(pattern, basename):
            return court_code

    return None


def detect_doc_type(pdf_path: str) -> str:
    """Detect document type from path.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        "order" or "complaint"
    """
    path_lower = pdf_path.lower()

    if "order" in path_lower:
        return "order"
    elif "complaint" in path_lower or "compliant" in path_lower:
        # Note: "compliant" handles the typo in "Selected Cases/Compliants"
        return "complaint"
    else:
        # Default to order (more common in this project)
        return "order"


def get_cleaner(
    pdf_path: str,
    doc_type: Optional[str] = None,
    court: Optional[str] = None,
    **kwargs
) -> BasePDFCleaner:
    """Get the appropriate cleaner for a PDF file.

    Auto-detects court from filename and document type from path if not provided.

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type ("order" or "complaint"), auto-detected if None
        court: Court code (e.g., "cand"), auto-detected if None
        **kwargs: Additional arguments passed to cleaner constructor

    Returns:
        Appropriate cleaner instance for the document
    """
    # Auto-detect court if not provided
    if court is None:
        court = detect_court(pdf_path)

    # Auto-detect document type if not provided
    if doc_type is None:
        doc_type = detect_doc_type(pdf_path)

    # Get the appropriate cleaner class
    if doc_type == "order":
        cleaner_class = ORDER_CLEANERS.get(court, GenericOrderCleaner)
    else:
        cleaner_class = COMPLAINT_CLEANERS.get(court, GenericComplaintCleaner)

    return cleaner_class(**kwargs)


def process_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    doc_type: Optional[str] = None,
    court: Optional[str] = None,
    save_chunks: bool = False,
    **kwargs
) -> Tuple[str, list]:
    """Process a PDF file with auto-detection of court and document type.

    Convenience function that combines get_cleaner() and process_pdf().

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save processed text (optional)
        doc_type: Document type ("order" or "complaint"), auto-detected if None
        court: Court code (e.g., "cand"), auto-detected if None
        save_chunks: If True, also save chunk files
        **kwargs: Additional arguments passed to cleaner constructor

    Returns:
        Tuple of (cleaned_text, chunks)
    """
    cleaner = get_cleaner(pdf_path, doc_type=doc_type, court=court, **kwargs)
    return cleaner.process_pdf(pdf_path, output_dir=output_dir, save_chunks=save_chunks)


def process_orders(
    input_dir: str = "Selected Cases/Orders/PDFs",
    output_dir: str = "Processed_Text/Orders_Cleaned"
) -> None:
    """Process all Order PDFs in a directory.

    Args:
        input_dir: Directory containing Order PDFs
        output_dir: Directory to save cleaned text files
    """
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} Order PDFs to process")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            process_pdf(pdf_path, output_dir, doc_type="order")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print(f"\nDone! Processed {len(pdf_files)} files to {output_dir}")


def process_complaints(
    input_dir: str = "Selected Cases/Compliants",
    output_dir: str = "Processed_Text/Complaints"
) -> None:
    """Process all Complaint PDFs in a directory.

    Args:
        input_dir: Directory containing Complaint PDFs
        output_dir: Directory to save cleaned text files
    """
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} Complaint PDFs to process")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            process_pdf(pdf_path, output_dir, doc_type="complaint")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print(f"\nDone! Processed {len(pdf_files)} files to {output_dir}")
