#!/bin/bash
# LaTeX reproduction script - simplified version
# 
# This script provides a simplified approach to document compilation.
# Use LATEX_OPTS to pass any additional options to latexmk.
# Command-line options will override any corresponding settings in .latexmkrc files.
# 
# Examples:
#   ./reproduce/reproduce_document_pdfs.sh                                        # Use .latexmkrc defaults
#   LATEX_OPTS="-interaction=batchmode" ./reproduce/reproduce_document_pdfs.sh   # Override interaction mode
#   LATEX_OPTS="-quiet -interaction=nonstopmode" ./reproduce/reproduce_document_pdfs.sh  # Multiple options
#
# To process only a single file, use latexmk directly:
#   latexmk document.tex                                  # Single file with .latexmkrc settings
#   LATEX_OPTS="-interaction=batchmode" latexmk document.tex     # Single file with options

# Check if latexmk is available
if ! command -v latexmk >/dev/null 2>&1; then
    echo "ERROR: latexmk is not installed or not in PATH"
    echo ""
    echo "latexmk is required for this script to work. Please install it:"
    echo "  - On macOS: brew install latexmk"
    echo "  - On Ubuntu/Debian: apt-get install latexmk"  
    echo "  - On other systems: install via your package manager or from CTAN"
    echo ""
    echo "For more information: https://ctan.org/pkg/latexmk"
    exit 1
fi

# Change to parent directory where the .tex files are located
cd ..

# Use LATEX_OPTS for any additional latexmk options
LATEXMK_OPTS="${LATEX_OPTS:-}"

# Triple compilation ensures all cross-references, citations, and table of contents are properly resolved
echo "Starting document compilation with triple latexmk run..."

latexmk -c
latexmk $LATEXMK_OPTS
latexmk $LATEXMK_OPTS  
latexmk $LATEXMK_OPTS

echo "Document compilation completed."
