#!/bin/bash

# This script contains all commands executed during the PDF build process
# 
# To run this script, ensure you have:
# - latexmk installed
# - All necessary .bib files in the current directory
# - All .tex files and dependencies in the current directory
#
# Usage: ./reproduce_document_pdf.sh [--content=all|body]
#
# Options:
#   --content=all   Compile all .tex files in the root directory (default)
#   --content=body  Compile only the main body and appendix files
#
# Examples:
#   ./reproduce_document_pdf.sh
#   ./reproduce_document_pdf.sh --content=all
#   ./reproduce_document_pdf.sh --content=body

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

CONTENT_MODE="all"  # Default to compiling everything

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --content=*)
            CONTENT_MODE="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--content=all|body]"
            echo ""
            echo "Options:"
            echo "  --content=all   Compile all .tex files in the root directory (default)"
            echo "  --content=body  Compile only the main body and appendix files"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate content mode
if [[ "$CONTENT_MODE" != "all" && "$CONTENT_MODE" != "body" ]]; then
    echo "Error: --content must be 'all' or 'body', got: $CONTENT_MODE"
    exit 1
fi

echo "Content mode: $CONTENT_MODE"

# =============================================================================
# SMART EXECUTION FUNCTIONS
# =============================================================================

# Function to get all .tex files to compile based on content mode
get_tex_files_to_compile() {
    case "$CONTENT_MODE" in
        "all")
            # Get ALL .tex files in the directory, excluding dotfiles
            find . -maxdepth 1 -name "*.tex" -type f 2>/dev/null | grep -v '/\.' | sort
            ;;
        "body")
            # Get main body file and any files containing "appendix" (case-insensitive)
            local files=()
            # Add main body file if it exists
            [[ -f "HAFiscal.tex" ]] && files+=("./HAFiscal.tex")
            # Add appendix files, excluding dotfiles
            local appendix_files=($(find . -maxdepth 1 -name "*.tex" -type f 2>/dev/null | grep -v '/\.' | grep -i appendix))
            files+=("${appendix_files[@]}")
            printf "%s\n" "${files[@]}" | sort
            ;;
    esac
}

# Function to compile documents based on content mode
compile_documents() {
    local tex_files=($(get_tex_files_to_compile))
    
    if [[ ${#tex_files[@]} -eq 0 ]]; then
        echo "No .tex files found for content mode: $CONTENT_MODE"
        return 1
    fi
    
    echo "Content mode: $CONTENT_MODE"
    echo "Files to compile: ${#tex_files[@]}"
    for file in "${tex_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    
    # Initial cleanup
    echo "Initial cleanup..."
    latexmk -c 2>/dev/null || true
    
    # Compile each file
    for tex_file in "${tex_files[@]}"; do
        local basename_file=$(basename "$tex_file" .tex)
        echo "=== Compiling $tex_file ==="
        
        # First pass: Generate initial .aux files
        echo "Pass 1: Generating .aux files"
        pdflatex -interaction=nonstopmode "$tex_file"
        
        # Process bibliography if .aux file was created
        if [[ -f "$basename_file.aux" ]]; then
            echo "Processing bibliography for $basename_file"
            bibtex "$basename_file.aux" 2>/dev/null || true
        fi
        
        # Second pass: Resolve citations
        echo "Pass 2: Resolving citations"
        pdflatex -interaction=nonstopmode "$tex_file"
        
        # Third pass: Final cross-reference resolution
        echo "Pass 3: Final resolution"
        pdflatex -interaction=nonstopmode "$tex_file"
        
        echo "Completed: $tex_file"
        echo ""
    done
    
    # Final cleanup
    echo "Final cleanup..."
    latexmk -c -r .latexmkrc-for-pdf 2>/dev/null || find . -maxdepth 1 \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' \) -delete
}

# =============================================================================
# REPRODUCTION COMMANDS
# =============================================================================

echo "Starting PDF compilation..."

# Execute compilation based on content mode
compile_documents latexmk -c 2>/dev/null || true
latexmk -f HAFiscal.tex
latexmk -f HAFiscal-online-appendix.tex
latexmk -f HAFiscal.tex
