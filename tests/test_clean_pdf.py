"""Tests for the clean_pdf package."""

import os
import pytest

from clean_pdf import (
    get_cleaner,
    detect_court,
    detect_doc_type,
    process_pdf,
    BackgroundExtractor,
    CANDOrderCleaner,
    NYSDOrderCleaner,
    GenericOrderCleaner,
    BasePDFCleaner,
    BaseOrderCleaner,
)


class TestCourtDetection:
    """Tests for court detection from filenames."""

    def test_detect_cand_underscore(self):
        assert detect_court("cand_22_cv_02094.pdf") == "cand"

    def test_detect_cand_dash(self):
        assert detect_court("cand-3_18-cv-04865.pdf") == "cand"

    def test_detect_nysd_underscore(self):
        assert detect_court("nysd_20_cv_04494.pdf") == "nysd"

    def test_detect_nysd_dash(self):
        assert detect_court("nysd-1_21-cv-11222.pdf") == "nysd"

    def test_detect_casd(self):
        assert detect_court("casd-3_23-cv-01216.pdf") == "casd"

    def test_detect_dcd(self):
        assert detect_court("dcd-1_23-cv-01599.pdf") == "dcd"

    def test_detect_ilnd(self):
        assert detect_court("ilnd-1_20_cv_05593.pdf") == "ilnd"

    def test_detect_azd(self):
        assert detect_court("azd-2_22-cv-02126.pdf") == "azd"

    def test_detect_unknown(self):
        assert detect_court("unknown_file.pdf") is None

    def test_detect_with_path(self):
        assert detect_court("/some/path/to/cand_22_cv_02094.pdf") == "cand"


class TestDocTypeDetection:
    """Tests for document type detection from paths."""

    def test_detect_order_in_path(self):
        assert detect_doc_type("Selected Cases/Orders/PDFs/cand_22_cv_02094.pdf") == "order"

    def test_detect_complaint_in_path(self):
        assert detect_doc_type("Selected Cases/Compliants/some_file.pdf") == "complaint"

    def test_default_to_order(self):
        assert detect_doc_type("some/random/path.pdf") == "order"


class TestCleanerFactory:
    """Tests for the get_cleaner factory function."""

    def test_get_cand_order_cleaner(self):
        cleaner = get_cleaner("cand_22_cv_02094.pdf", doc_type="order")
        assert isinstance(cleaner, CANDOrderCleaner)

    def test_get_nysd_order_cleaner(self):
        cleaner = get_cleaner("nysd_20_cv_04494.pdf", doc_type="order")
        assert isinstance(cleaner, NYSDOrderCleaner)

    def test_get_generic_for_unknown(self):
        cleaner = get_cleaner("unknown.pdf", doc_type="order")
        assert isinstance(cleaner, GenericOrderCleaner)

    def test_get_generic_for_azd(self):
        cleaner = get_cleaner("azd-2_22-cv-02126.pdf", doc_type="order")
        assert isinstance(cleaner, GenericOrderCleaner)

    def test_auto_detect_court(self):
        cleaner = get_cleaner("Selected Cases/Orders/PDFs/cand_22_cv_02094.pdf")
        assert isinstance(cleaner, CANDOrderCleaner)

    def test_explicit_court_overrides_detection(self):
        cleaner = get_cleaner("cand_file.pdf", doc_type="order", court="nysd")
        assert isinstance(cleaner, NYSDOrderCleaner)


class TestBackgroundExtractor:
    """Tests for the BackgroundExtractor class."""

    def test_extract_roman_numeral_background(self):
        text = "Some intro text. I. BACKGROUND This is the background section with facts. II. LEGAL STANDARD The legal standard."
        extractor = BackgroundExtractor()
        result = extractor.extract(text)
        assert result is not None
        assert "background section with facts" in result
        assert "LEGAL STANDARD" not in result

    def test_extract_plain_background(self):
        text = "Caption here. BACKGROUND The facts of the case. DISCUSSION Analysis begins."
        extractor = BackgroundExtractor()
        result = extractor.extract(text)
        assert result is not None
        assert "facts of the case" in result

    def test_extract_factual_background(self):
        text = "Intro. I. FACTUAL BACKGROUND The detailed facts. II. DISCUSSION The analysis."
        extractor = BackgroundExtractor()
        result = extractor.extract(text)
        assert result is not None
        assert "detailed facts" in result

    def test_extract_facts_section(self):
        text = "Caption. FACTS These are the facts. LEGAL STANDARD The standard."
        extractor = BackgroundExtractor()
        result = extractor.extract(text)
        assert result is not None
        assert "These are the facts" in result

    def test_no_background_section(self):
        text = "This document has no background section, just random text about legal matters."
        extractor = BackgroundExtractor()
        result = extractor.extract(text)
        assert result is None

    def test_has_background_section(self):
        extractor = BackgroundExtractor()
        assert extractor.has_background_section("Some text I. BACKGROUND more text")
        assert not extractor.has_background_section("No section headers here")

    def test_get_section_header(self):
        extractor = BackgroundExtractor()
        text = "Intro. I. FACTUAL BACKGROUND The facts."
        header = extractor.get_section_header(text)
        assert header is not None
        assert "FACTUAL BACKGROUND" in header

    def test_include_header_option(self):
        text = "Caption. I. BACKGROUND The facts here. II. DISCUSSION Analysis."
        extractor = BackgroundExtractor()

        without_header = extractor.extract(text, include_header=False)
        with_header = extractor.extract(text, include_header=True)

        assert "I. BACKGROUND" not in without_header
        assert "I. BACKGROUND" in with_header

    def test_get_boundaries(self):
        text = "Caption. I. BACKGROUND The facts. II. DISCUSSION Analysis."
        extractor = BackgroundExtractor()
        boundaries = extractor.get_boundaries(text)

        assert boundaries is not None
        start, end = boundaries
        assert start < end
        assert text[start:end].strip().startswith("I. BACKGROUND")


class TestBasePDFCleaner:
    """Tests for base cleaner functionality."""

    def test_fix_ocr_errors(self):
        cleaner = GenericOrderCleaner()
        # Test ligature fixes
        assert "fi" in cleaner._fix_ocr_errors("fi")
        assert "fl" in cleaner._fix_ocr_errors("fl")

    def test_remove_margin_numbers(self):
        cleaner = GenericOrderCleaner()
        assert cleaner._remove_margin_numbers("15 This is content") == "This is content"
        assert cleaner._remove_margin_numbers("1 First line") == "First line"
        assert cleaner._remove_margin_numbers("28 Last margin number") == "Last margin number"
        # Should not remove numbers outside 1-28 range
        assert cleaner._remove_margin_numbers("50 Not a margin") == "50 Not a margin"
        # Should not remove numbered list items
        assert cleaner._remove_margin_numbers("1. List item") == "1. List item"


class TestCANDCleaner:
    """Tests for CAND-specific cleaner functionality."""

    def test_garbled_line_detection(self):
        cleaner = CANDOrderCleaner()
        assert cleaner._is_garbled_line("u o r o 13")
        assert cleaner._is_garbled_line("a t r i n")
        assert cleaner._is_garbled_line("cC 14")
        assert not cleaner._is_garbled_line("This is normal text")

    def test_remove_garbled_text(self):
        cleaner = CANDOrderCleaner()
        text = "The court finds that u o r o the defendant is liable."
        result = cleaner._remove_garbled_text(text)
        assert "u o r o" not in result
        assert "defendant is liable" in result


class TestIntegration:
    """Integration tests with actual PDF files."""

    @pytest.fixture
    def pdf_dir(self):
        return "Selected Cases/Orders/PDFs"

    def test_process_cand_pdf(self, pdf_dir):
        pdf_path = os.path.join(pdf_dir, "cand_22_cv_02094.pdf")
        if not os.path.exists(pdf_path):
            pytest.skip("Test PDF not found")

        text, chunks = process_pdf(pdf_path)

        assert len(text) > 0
        assert len(chunks) > 0
        assert "Lucid" in text  # Known content from this document

    def test_process_nysd_pdf(self, pdf_dir):
        pdf_path = os.path.join(pdf_dir, "nysd_20_cv_04494.pdf")
        if not os.path.exists(pdf_path):
            pytest.skip("Test PDF not found")

        text, chunks = process_pdf(pdf_path)

        assert len(text) > 0
        assert len(chunks) > 0

    def test_background_extraction_from_pdf(self, pdf_dir):
        pdf_path = os.path.join(pdf_dir, "cand_22_cv_02094.pdf")
        if not os.path.exists(pdf_path):
            pytest.skip("Test PDF not found")

        text, _ = process_pdf(pdf_path)
        extractor = BackgroundExtractor()
        background = extractor.extract(text)

        assert background is not None
        assert len(background) > 100

    def test_all_pdfs_processable(self, pdf_dir):
        """Verify all PDFs in the directory can be processed without errors."""
        if not os.path.exists(pdf_dir):
            pytest.skip("PDF directory not found")

        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            pytest.skip("No PDF files found")

        errors = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                text, chunks = process_pdf(pdf_path)
                assert len(text) > 0, f"{pdf_file}: Empty text"
            except Exception as e:
                errors.append(f"{pdf_file}: {str(e)}")

        assert not errors, f"Errors processing PDFs: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
