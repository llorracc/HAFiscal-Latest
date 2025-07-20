#!/bin/bash

# CI Test Script for HAFiscal Repository
# This script is optimized for continuous integration testing
# It performs minimal but sufficient tests to verify repository functionality

set -e  # Exit on any error
set -o pipefail  # Exit if any command in a pipe fails

echo "=== HAFiscal CI Test ==="
echo "Starting CI test at $(date)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a file exists
file_exists() {
    [[ -f "$1" ]]
}

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check LaTeX compilation success
check_pdf_generated() {
    local tex_file="$1"
    local pdf_file="${tex_file%.tex}.pdf"
    
    if [[ -f "$pdf_file" ]]; then
        local file_size=$(stat -f%z "$pdf_file" 2>/dev/null || stat -c%s "$pdf_file" 2>/dev/null || echo "0")
        if [[ "$file_size" -gt 1000 ]]; then  # PDF should be at least 1KB
            log "✓ PDF generated successfully: $pdf_file ($file_size bytes)"
            return 0
        else
            log "✗ PDF file too small: $pdf_file ($file_size bytes)"
            return 1
        fi
    else
        log "✗ PDF file not found: $pdf_file"
        return 1
    fi
}

# Function to run LaTeX compilation with error checking
compile_tex_file() {
    local tex_file="$1"
    local basename_file=$(basename "$tex_file" .tex)
    
    log "Compiling $tex_file..."
    
    # Clean previous artifacts
    rm -f "$basename_file.aux" "$basename_file.log" "$basename_file.out" "$basename_file.toc" "$basename_file.bbl" "$basename_file.blg" 2>/dev/null || true
    
    # First pass: Generate initial .aux files
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$tex_file" > "${basename_file}_compile.log" 2>&1; then
        log "✗ First pdflatex pass failed for $tex_file"
        cat "${basename_file}_compile.log"
        return 1
    fi
    
    # Process bibliography if .aux file was created
    if [[ -f "$basename_file.aux" ]]; then
        log "Processing bibliography for $basename_file..."
        if ! bibtex "$basename_file" > "${basename_file}_bibtex.log" 2>&1; then
            log "⚠ Bibliography processing had issues for $basename_file (continuing anyway)"
        fi
    fi
    
    # Second pass: Resolve citations
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$tex_file" >> "${basename_file}_compile.log" 2>&1; then
        log "✗ Second pdflatex pass failed for $tex_file"
        cat "${basename_file}_compile.log"
        return 1
    fi
    
    # Third pass: Final cross-reference resolution
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$tex_file" >> "${basename_file}_compile.log" 2>&1; then
        log "✗ Third pdflatex pass failed for $tex_file"
        cat "${basename_file}_compile.log"
        return 1
    fi
    
    # Check if PDF was generated successfully
    if check_pdf_generated "$tex_file"; then
        log "✓ Successfully compiled $tex_file"
        return 0
    else
        log "✗ PDF generation failed for $tex_file"
        return 1
    fi
}

# =============================================================================
# MAIN CI TEST SEQUENCE
# =============================================================================

# Test 1: Check essential dependencies
log "Test 1: Checking dependencies..."
if ! command_exists pdflatex; then
    log "✗ pdflatex not found - LaTeX installation required"
    exit 1
fi

if ! command_exists bibtex; then
    log "✗ bibtex not found - LaTeX installation required"
    exit 1
fi

log "✓ LaTeX dependencies found"

# Test 2: Check essential files exist
log "Test 2: Checking essential files..."
required_files=(
    "HAFiscal.tex"
    "HAFiscal-online-appendix.tex"
    "references-paperpile.bib"
    "README.md"
)

for file in "${required_files[@]}"; do
    if file_exists "$file"; then
        log "✓ Found $file"
    else
        log "✗ Missing required file: $file"
        exit 1
    fi
done

# Test 3: Quick syntax check of main TeX files
log "Test 3: Checking TeX file syntax..."
for tex_file in "HAFiscal.tex" "HAFiscal-online-appendix.tex"; do
    if [[ -f "$tex_file" ]]; then
        # Quick syntax check - just parse without full compilation
        if pdflatex -interaction=nonstopmode -draftmode "$tex_file" > /dev/null 2>&1; then
            log "✓ Syntax check passed for $tex_file"
        else
            log "✗ Syntax check failed for $tex_file"
            exit 1
        fi
    fi
done

# Test 4: Compile main document (minimal version for CI)
log "Test 4: Compiling main document..."
if compile_tex_file "HAFiscal.tex"; then
    log "✓ Main document compilation successful"
else
    log "✗ Main document compilation failed"
    exit 1
fi

# Test 5: Compile appendix (if it exists and is different from main)
if [[ -f "HAFiscal-online-appendix.tex" ]]; then
    log "Test 5: Compiling appendix..."
    if compile_tex_file "HAFiscal-online-appendix.tex"; then
        log "✓ Appendix compilation successful"
    else
        log "✗ Appendix compilation failed"
        exit 1
    fi
else
    log "⚠ Appendix file not found, skipping Test 5"
fi

# Test 6: Check Python environment (if available)
if command_exists python; then
    log "Test 6: Checking Python environment..."
    python_version=$(python --version 2>&1 || echo "Unknown")
    log "✓ Python found: $python_version"
    
    # Check if key Python files exist
    if [[ -f "Code/HA-Models/do_all.py" ]]; then
        log "✓ Main Python script found"
    else
        log "⚠ Main Python script not found"
    fi
else
    log "⚠ Python not found, skipping Python environment check"
fi

# Test 7: Check repository structure
log "Test 7: Checking repository structure..."
required_dirs=("Code" "Figures" "Tables")
for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        log "✓ Found directory: $dir"
    else
        log "⚠ Missing directory: $dir"
    fi
done

# =============================================================================
# CLEANUP AND FINAL REPORT
# =============================================================================

# Clean up compilation artifacts
log "Cleaning up compilation artifacts..."
find . -maxdepth 1 \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' -o -name '*_compile.log' -o -name '*_bibtex.log' \) -delete 2>/dev/null || true

# Final success report
log "=== CI Test Results ==="
log "✓ All tests passed successfully!"
log "✓ Repository is ready for use"
log "✓ CI test completed at $(date)"

echo ""
echo "Generated PDFs:"
if [[ -f "HAFiscal.pdf" ]]; then
    echo "  - HAFiscal.pdf"
fi
if [[ -f "HAFiscal-online-appendix.pdf" ]]; then
    echo "  - HAFiscal-online-appendix.pdf"
fi

exit 0 