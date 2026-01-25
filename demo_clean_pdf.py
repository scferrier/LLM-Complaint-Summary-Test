#!/usr/bin/env python3
"""
Demo script for processing PDF documents using the clean_pdf utility.

This script demonstrates how to use the PDFTextProcessor class to:
1. Extract text from PDF files
2. Clean the text for LLM processing
3. Split large documents into manageable chunks
4. Save processed files for downstream use
"""

import os
import sys
from pathlib import Path
from clean_pdf import PDFTextProcessor


def process_single_pdf(pdf_path: str, output_dir: str = "processed_text"):
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save processed files
    """
    processor = PDFTextProcessor()

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False

    try:
        cleaned_text, chunks = processor.process_pdf(pdf_path, output_dir)

        print(f"\n✅ Successfully processed: {os.path.basename(pdf_path)}")
        print(
            f"   Original characters: {len(processor.extract_text_from_pdf(pdf_path))}"
        )
        print(f"   Cleaned characters: {len(cleaned_text)}")
        print(f"   Number of chunks: {len(chunks)}")
        print(f"   Files saved to: {output_dir}")

        # Show a preview
        print(f"\n📄 Preview of cleaned text:")
        print("-" * 50)
        print(cleaned_text[:400] + "..." if len(cleaned_text) > 400 else cleaned_text)
        print("-" * 50)

        return True

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return False


def process_directory(input_dir: str, output_base_dir: str = "processed_text"):
    """
    Process all PDF files in a directory.

    Args:
        input_dir: Directory containing PDF files
        output_base_dir: Base directory for output files
    """
    processor = PDFTextProcessor()

    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return

    print(f"📁 Processing {len(pdf_files)} PDF files from: {input_dir}")

    # Create category-specific output directory
    category_name = os.path.basename(input_dir).replace("/", "_")
    output_dir = os.path.join(output_base_dir, category_name)

    success_count = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        if process_single_pdf(pdf_path, output_dir):
            success_count += 1

    print(
        f"\n🎉 Processing complete! Successfully processed {success_count}/{len(pdf_files)} files."
    )


def demo_processing():
    """Demonstrate processing on sample PDF files."""
    print("🚀 PDF Processing Demo")
    print("=" * 50)

    # Process all PDFs in the Selected Cases directory
    base_dir = "Selected Cases"

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found.")
        print("Please ensure the PDF files are in the correct location.")
        return

    # Process complaints
    complaints_dir = os.path.join(base_dir, "Compliants")
    if os.path.exists(complaints_dir):
        print(f"\n📋 Processing Complaints...")
        process_directory(complaints_dir)

    # Process orders
    orders_dir = os.path.join(base_dir, "Orders/PDFs")
    if os.path.exists(orders_dir):
        print(f"\n⚖️ Processing Orders...")
        process_directory(orders_dir)

    print(f"\n✨ Demo complete! Check the 'processed_text' directory for results.")
    print(f"\n💡 Usage tips:")
    print(f"   - Use the cleaned .txt files for LLM input")
    print(f"   - Use the chunked files for very large documents")
    print(
        f"   - Text has been cleaned of headers, footers, page numbers, and signature blocks"
    )


def interactive_mode():
    """Interactive mode for processing specific files."""
    print("🔧 Interactive PDF Processing")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("1. Process a single PDF file")
        print("2. Process a directory of PDFs")
        print("3. Run demo on Selected Cases")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            pdf_path = input("Enter path to PDF file: ").strip()
            output_dir = input(
                "Enter output directory (default: processed_text): "
            ).strip()
            if not output_dir:
                output_dir = "processed_text"
            process_single_pdf(pdf_path, output_dir)

        elif choice == "2":
            input_dir = input("Enter path to directory containing PDFs: ").strip()
            output_dir = input(
                "Enter base output directory (default: processed_text): "
            ).strip()
            if not output_dir:
                output_dir = "processed_text"
            process_directory(input_dir, output_dir)

        elif choice == "3":
            demo_processing()

        elif choice == "4":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_processing()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        elif sys.argv[1] == "--help":
            print("PDF Processing Utility")
            print("Usage:")
            print(
                "  python demo_clean_pdf.py --demo       # Run demo on Selected Cases"
            )
            print("  python demo_clean_pdf.py --interactive # Interactive mode")
            print("  python demo_clean_pdf.py --help        # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default to demo mode
        demo_processing()


if __name__ == "__main__":
    main()
