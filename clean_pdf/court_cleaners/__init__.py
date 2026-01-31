"""Court-specific PDF cleaners for different district courts."""

from .cand import CANDOrderCleaner, CANDComplaintCleaner
from .nysd import NYSDOrderCleaner, NYSDComplaintCleaner
from .casd import CASDOrderCleaner, CASDComplaintCleaner
from .dcd import DCDOrderCleaner, DCDComplaintCleaner
from .ilnd import ILNDOrderCleaner, ILNDComplaintCleaner
from .generic import GenericOrderCleaner, GenericComplaintCleaner

__all__ = [
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
]
