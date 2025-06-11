#!/bin/bash
# HAFiscal body reproduction script
# 
# Setup LaTeX compilation environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/reproduce/setup-latexmk.sh"
source "$SCRIPT_DIR/reproduce/setup-latex-interaction.sh"

# Set interaction mode for LaTeX compilation
# Environment variable LATEX_INTERACTION_MODE can override the default
setup_latex_interaction

# latexmk compiles .tex documents with automatic handling of bibliography and cross-references
# Documentation: https://ctan.org/pkg/latexmk
latexmk -c 2>/dev/null || true
latexmk  "HAFiscal-online-appendix.tex"
latexmk  "HAFiscal.tex"
latexmk -c 2>/dev/null || true
