#!/bin/bash

# Shared library for PDF document reproduction scripts
# This file contains common functions used by both reproduce_document_pdf.sh and reproduce_document_pdf_ci.sh

# =============================================================================
# SHARED FUNCTIONS
# =============================================================================

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a file exists
file_exists() {
    [[ -f "$1" ]]
}

# Function to log with timestamp (only in CI mode)
log() {
    if [[ "${CI_MODE:-false}" == "true" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    else
        echo "$1"
    fi
}

# Function to check if PDF was generated successfully
check_pdf_success() {
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
        log "No .tex files found for content mode: $CONTENT_MODE"
        return 1
    fi
    
    log "Content mode: $CONTENT_MODE"
    log "Files to compile: ${#tex_files[@]}"
    for file in "${tex_files[@]}"; do
        log "  - $file"
    done
    echo ""
    
    # Initial cleanup
    log "Initial cleanup..."
    latexmk -c 2>/dev/null || true
    
    # Compile each file
    for tex_file in "${tex_files[@]}"; do
        local basename_file=$(basename "$tex_file" .tex)
        log "=== Compiling $tex_file ==="
        
        # First pass: Generate initial .aux files
        log "Pass 1: Generating .aux files"
        if [[ "${CI_MODE:-false}" == "true" ]]; then
            if ! pdflatex -interaction=nonstopmode "$tex_file" > "${basename_file}_compile.log" 2>&1; then
                log "✗ First pdflatex pass failed for $tex_file"
                cat "${basename_file}_compile.log"
                return 1
            fi
        else
            pdflatex -interaction=nonstopmode "$tex_file"
        fi
        
        # Process bibliography if .aux file was created
        if [[ -f "$basename_file.aux" ]]; then
            log "Processing bibliography for $basename_file"
            if [[ "${CI_MODE:-false}" == "true" ]]; then
                if ! bibtex "$basename_file.aux" > "${basename_file}_bibtex.log" 2>&1; then
                    log "⚠ Bibliography processing had issues for $basename_file (continuing anyway)"
                fi
            else
                bibtex "$basename_file.aux" 2>/dev/null || true
            fi
        fi
        
        # Second pass: Resolve citations
        log "Pass 2: Resolving citations"
        if [[ "${CI_MODE:-false}" == "true" ]]; then
            if ! pdflatex -interaction=nonstopmode "$tex_file" >> "${basename_file}_compile.log" 2>&1; then
                log "✗ Second pdflatex pass failed for $tex_file"
                cat "${basename_file}_compile.log"
                return 1
            fi
        else
            pdflatex -interaction=nonstopmode "$tex_file"
        fi
        
        # Third pass: Final cross-reference resolution
        log "Pass 3: Final resolution"
        if [[ "${CI_MODE:-false}" == "true" ]]; then
            if ! pdflatex -interaction=nonstopmode "$tex_file" >> "${basename_file}_compile.log" 2>&1; then
                log "✗ Third pdflatex pass failed for $tex_file"
                cat "${basename_file}_compile.log"
                return 1
            fi
        else
            pdflatex -interaction=nonstopmode "$tex_file"
        fi
        
        # Check PDF generation in CI mode
        if [[ "${CI_MODE:-false}" == "true" ]]; then
            if check_pdf_success "$tex_file"; then
                log "✓ Successfully compiled $tex_file"
            else
                log "✗ PDF generation failed for $tex_file"
                return 1
            fi
        else
            log "Completed: $tex_file"
        fi
        echo ""
    done
    
    # Final cleanup
    log "Final cleanup..."
    latexmk -c -r .latexmkrc-for-pdf 2>/dev/null || find . -maxdepth 1 \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' \) -delete
}

# Function to build with latexmk (with CI error handling)
build_with_latexmk() {
    local tex_file="$1"
    local log_file="${tex_file%.tex}_build.log"
    
    if [[ "${CI_MODE:-false}" == "true" ]]; then
        if ! latexmk -f "$tex_file" 2>&1 | tee "$log_file"; then
            # Check if PDF was actually generated despite warnings
            if [[ -f "${tex_file%.tex}.pdf" ]]; then
                local file_size=$(stat -f%z "${tex_file%.tex}.pdf" 2>/dev/null || stat -c%s "${tex_file%.tex}.pdf" 2>/dev/null || echo "0")
                if [[ "$file_size" -gt 1000 ]]; then
                    log "⚠ $tex_file build had warnings but PDF was generated successfully"
                else
                    log "✗ $tex_file build failed - PDF too small"
                    return 1
                fi
            else
                log "✗ $tex_file build failed - no PDF generated"
                return 1
            fi
        fi
    else
        latexmk -f "$tex_file"
    fi
}

# Main reproduction function
reproduce_commands() {
    log "Starting PDF compilation..."

    if $HAVE_LATEXMK; then
        log "latexmk detected – using fast build path"
        echo ""
        # Clean previous artifacts
        latexmk -C >/dev/null 2>&1 || true

        # Suppress cross-doc warning in .latexmkrc
        export DO_NOT_WARN=1

        # Build appendix first, then main paper (two passes for safety)
        log "Building appendix..."
        build_with_latexmk "HAFiscal-online-appendix.tex"
        
        log "Building main document..."
        build_with_latexmk "HAFiscal.tex"
        
        log "Final passes for cross-references..."
        latexmk -f HAFiscal-online-appendix.tex
        latexmk -f HAFiscal.tex   # extra pass to polish x-refs
        latexmk -c 
    else
        log "latexmk NOT found – falling back to manual pdflatex/bibtex cycle"
        echo ""
        compile_documents
    fi
    
    # Final verification in CI mode
    if [[ "${CI_MODE:-false}" == "true" ]]; then
        log "Verifying generated PDFs..."
        local success=true
        
        if [[ -f "HAFiscal.tex" ]]; then
            if ! check_pdf_success "HAFiscal.tex"; then
                success=false
            fi
        fi
        
        if [[ -f "HAFiscal-online-appendix.tex" ]]; then
            if ! check_pdf_success "HAFiscal-online-appendix.tex"; then
                success=false
            fi
        fi
        
        if $success; then
            log "✓ All PDFs generated successfully"
        else
            log "✗ Some PDFs failed to generate"
            return 1
        fi
    fi
} 