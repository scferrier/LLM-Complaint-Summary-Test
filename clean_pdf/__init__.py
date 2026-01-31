"""
Clean PDF - Modular PDF text extraction and cleaning for legal documents.

This package provides court-specific cleaners for extracting and cleaning
text from legal PDF documents (complaints and orders) for LLM processing.

Usage:
    from clean_pdf import process_pdf, get_cleaner, BackgroundExtractor

    # Process with auto-detection of court and document type
    text, chunks = process_pdf("path/to/cand_22_cv_02094.pdf", output_dir="output/")

    # Get a specific cleaner
    cleaner = get_cleaner("path/to/nysd_20_cv_04494.pdf", doc_type="order")
    text, chunks = cleaner.process_pdf("path/to/nysd_20_cv_04494.pdf")

    # Extract background section from cleaned text
    extractor = BackgroundExtractor()
    background = extractor.extract(text)
"""

# Base classes
from .base import BasePDFCleaner
from .order_base import BaseOrderCleaner
from .complaint_base import BaseComplaintCleaner

# Factory functions
from .factory import (
    get_cleaner,
    process_pdf,
    process_orders,
    process_complaints,
    detect_court,
    detect_doc_type,
)

# Court-specific cleaners
from .court_cleaners import (
    # Order cleaners
    CANDOrderCleaner,
    NYSDOrderCleaner,
    CASDOrderCleaner,
    DCDOrderCleaner,
    ILNDOrderCleaner,
    GenericOrderCleaner,
    # Complaint cleaners
    CANDComplaintCleaner,
    NYSDComplaintCleaner,
    CASDComplaintCleaner,
    DCDComplaintCleaner,
    ILNDComplaintCleaner,
    GenericComplaintCleaner,
)

# Extractors
from .extractors import BackgroundExtractor

__all__ = [
    # Base classes
    "BasePDFCleaner",
    "BaseOrderCleaner",
    "BaseComplaintCleaner",
    # Factory functions
    "get_cleaner",
    "process_pdf",
    "process_orders",
    "process_complaints",
    "detect_court",
    "detect_doc_type",
    # Order cleaners
    "CANDOrderCleaner",
    "NYSDOrderCleaner",
    "CASDOrderCleaner",
    "DCDOrderCleaner",
    "ILNDOrderCleaner",
    "GenericOrderCleaner",
    # Complaint cleaners
    "CANDComplaintCleaner",
    "NYSDComplaintCleaner",
    "CASDComplaintCleaner",
    "DCDComplaintCleaner",
    "ILNDComplaintCleaner",
    "GenericComplaintCleaner",
    # Extractors
    "BackgroundExtractor",
]

__version__ = "1.0.0"
