#!/bin/bash

# This script contains all commands executed during the PDF build process
# CI-FRIENDLY VERSION with proper error handling and exit codes
# 
# To run this script, ensure you have:
# - latexmk installed
# - All necessary .bib files in the current directory
# - All .tex files and dependencies in the current directory
#
# Usage: ./reproduce_document_pdf_ci.sh [--content=all|body] [--ci-mode]
#
# Options:
#   --content=all   Compile all .tex files in the root directory (default)
#   --content=body  Compile only the main body and appendix files
#   --ci-mode       Enable CI mode with stricter error checking
#
# Examples:
#   ./reproduce_document_pdf_ci.sh
#   ./reproduce_document_pdf_ci.sh --content=all
#   ./reproduce_document_pdf_ci.sh --content=body --ci-mode

# CI mode settings
CI_MODE=false
set -e  # Exit on any error (only in CI mode)

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
        --ci-mode)
            CI_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--content=all|body] [--ci-mode]"
            echo ""
            echo "Options:"
            echo "  --content=all   Compile all .tex files in the root directory (default)"
            echo "  --content=body  Compile only the main body and appendix files"
            echo "  --ci-mode       Enable CI mode with stricter error checking"
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
if $CI_MODE; then
    echo "CI mode: enabled"
    set -e  # Exit on any error
    set -o pipefail  # Exit if any command in a pipe fails
fi

# Detect whether latexmk is available on this system
if command -v latexmk >/dev/null 2>&1; then
    HAVE_LATEXMK=true
else
    HAVE_LATEXMK=false
fi

# =============================================================================
# LOAD SHARED LIBRARY
# =============================================================================

# Source the shared library (same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/reproduce_document_pdf_lib.sh"

# =============================================================================
# Execute the build
# =============================================================================
if reproduce_commands; then
    if $CI_MODE; then
        log "=== CI Test Results ==="
        log "✓ PDF compilation successful!"
        log "✓ Repository is ready for use"
    fi
    exit 0
else
    if $CI_MODE; then
        log "=== CI Test Results ==="
        log "✗ PDF compilation failed!"
        log "✗ Repository needs attention"
    fi
    exit 1
fi 