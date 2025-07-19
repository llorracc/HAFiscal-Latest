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

# Get the directory of this script and change to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Use LATEX_OPTS for any additional latexmk options
LATEXMK_OPTS="${LATEX_OPTS:-}"

# Define default files to compile if none are specified
DEFAULT_FILES="HAFiscal.tex HAFiscal-online-appendix.tex"

# Triple compilation ensures all cross-references, citations, and table of contents are properly resolved
echo "Starting document compilation with triple latexmk run..."

latexmk -c
# Run latexmk on each document with multiple passes to resolve cross-references
for doc in $DEFAULT_FILES; do
    if [[ -f "$doc" ]]; then
        echo "Compiling $doc..."
        latexmk $LATEXMK_OPTS "$doc"
    else
        echo "Warning: $doc not found, skipping..."
    fi
done

# Second pass to resolve cross-document references
for doc in $DEFAULT_FILES; do
    if [[ -f "$doc" ]]; then
        echo "Second pass for $doc..."
        latexmk $LATEXMK_OPTS "$doc"
    fi
done

# Third pass to finalize all references
for doc in $DEFAULT_FILES; do
    if [[ -f "$doc" ]]; then
        echo "Final pass for $doc..."
        latexmk $LATEXMK_OPTS "$doc"
    fi
done

echo "Document compilation completed."
