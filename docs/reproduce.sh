#!/bin/bash

# HAFiscal Reproduction Script
# This script provides options for reproducing different aspects of the HAFiscal project

set -eo pipefail

show_help() {
    cat << EOF
HAFiscal Reproduction Script

This script provides multiple reproduction options:

USAGE:
    ./reproduce.sh [OPTION]

OPTIONS:
    --help, -h          Show this help message
    --docs, -d          Reproduce LaTeX documents
    --all, -a           Reproduce all computational results (may take 1-2 days)
    --min, -m           Reproduce minimal computational results (~1 hour)
    --interactive, -i   Show interactive menu (default when run from terminal)

ENVIRONMENT VARIABLES:
    REPRODUCE_TARGETS   Comma-separated list of targets to reproduce (non-interactive mode)
                       Valid values: docs, min, all
                       Examples:
                         REPRODUCE_TARGETS=docs
                         REPRODUCE_TARGETS=min,docs  
                         REPRODUCE_TARGETS=all

EXAMPLES:
    ./reproduce.sh                           # Interactive menu (if terminal)
    ./reproduce.sh --docs                    # Just compile documents
    ./reproduce.sh --min                     # Quick computational reproduction
    ./reproduce.sh --all                     # Full computational reproduction
    
    # Non-interactive examples:
    REPRODUCE_TARGETS=docs ./reproduce.sh    # Documents only
    REPRODUCE_TARGETS=min,docs ./reproduce.sh # Minimal results + documents
    echo | REPRODUCE_TARGETS=min ./reproduce.sh # Force non-interactive

EOF
}

show_interactive_menu() {
    echo "========================================"
    echo "   HAFiscal Reproduction Options"
    echo "========================================"
    echo ""
    echo "Please select what you would like to reproduce:"
    echo ""
    echo "1) LaTeX Documents"
    echo "   - Compiles all PDF documents from LaTeX source"
    echo "   - Estimated time: A few minutes"
    echo ""
    echo "2) All Computational Results"
    echo "   - Reproduces all computational results from the paper"
    echo "   - ⚠️  WARNING: This may take 1-2 DAYS to complete"
    echo "   - Requires significant computational resources"
    echo ""
    echo "3) Minimal Computational Results"
    echo "   - Reproduces a subset of computational results"
    echo "   - Estimated time: ~1 hour"
    echo "   - Good for testing and quick verification"
    echo ""
    echo "4) Exit"
    echo ""
    echo -n "Enter your choice (1-4): "
}

reproduce_documents() {
    echo "========================================"
    echo "Reproducing LaTeX Documents..."
    echo "========================================"
    echo ""
    
    if [[ -f "./reproduce/reproduce_document_pdfs.sh" ]]; then
        ./reproduce/reproduce_document_pdfs.sh
    else
        echo "ERROR: ./reproduce/reproduce_document_pdfs.sh not found"
        echo "Please run the document PDF maker script first"
        return 1
    fi
}

reproduce_all_results() {
    echo "========================================"
    echo "Reproducing All Computational Results..."
    echo "========================================"
    echo ""
    echo "⚠️  WARNING: This process may take 1-2 DAYS to complete!"
    echo "Make sure you have:"
    echo "- Sufficient computational resources"
    echo "- Stable power supply"
    echo "- No other intensive processes running"
    echo ""
    
    if is_interactive; then
        echo -n "Are you sure you want to continue? (y/N): "
        read -r confirm
        
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            echo ""
            echo "Starting full computational reproduction..."
        else
            echo "Cancelled by user."
            return 0
        fi
    else
        echo "Running in non-interactive mode - proceeding with full reproduction..."
        echo ""
    fi
    
    if [[ -f "./reproduce/reproduce_computed.sh" ]]; then
        ./reproduce/reproduce_computed.sh
    else
        echo "ERROR: ./reproduce/reproduce_computed.sh not found"
        return 1
    fi
}

reproduce_minimal_results() {
    echo "========================================"
    echo "Reproducing Minimal Computational Results..."
    echo "========================================"
    echo ""
    echo "This will reproduce a subset of results (~1 hour)"
    echo ""
    
    if [[ -f "./reproduce/reproduce_computed_min.sh" ]]; then
        ./reproduce/reproduce_computed_min.sh
    else
        echo "ERROR: ./reproduce/reproduce_computed_min.sh not found"
        return 1
    fi
}

run_interactive_menu() {
    while true; do
        show_interactive_menu
        read -r choice
        echo ""
        
        case $choice in
            1)
                reproduce_documents
                break
                ;;
            2)
                reproduce_all_results
                break
                ;;
            3)
                reproduce_minimal_results
                break
                ;;
            4)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid choice. Please enter 1, 2, 3, or 4."
                echo ""
                ;;
        esac
    done
}

is_interactive() {
    # Check if both stdin and stdout are terminals
    [[ -t 0 && -t 1 ]]
}

process_reproduce_targets() {
    local targets="${REPRODUCE_TARGETS:-}"
    
    if [[ -z "$targets" ]]; then
        echo "ERROR: REPRODUCE_TARGETS environment variable not set"
        echo "Valid values: docs, min, all (comma-separated)"
        echo "Example: REPRODUCE_TARGETS=docs,min"
        return 1
    fi
    
    # Replace commas with spaces for simple iteration
    local targets_spaced=$(echo "$targets" | tr ',' ' ')
    
    local has_error=false
    local executed_targets=""
    
    # Validate all targets first
    for target in $targets_spaced; do
        # Trim whitespace
        target=$(echo "$target" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        case "$target" in
            docs|min|all)
                # Valid target
                ;;
            *)
                echo "ERROR: Invalid target '$target'"
                echo "Valid targets: docs, min, all"
                has_error=true
                ;;
        esac
    done
    
    if [[ "$has_error" == true ]]; then
        return 1
    fi
    
    # Execute targets in a logical order: docs, min, all
    for ordered_target in docs min all; do
        for target in $targets_spaced; do
            target=$(echo "$target" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            if [[ "$target" == "$ordered_target" ]]; then
                # Check if we've already executed this target
                if [[ "$executed_targets" != *"$target"* ]]; then
                    echo "Executing target: $target"
                    case "$target" in
                        docs)
                            reproduce_documents || return 1
                            ;;
                        min)
                            reproduce_minimal_results || return 1
                            ;;
                        all)
                            reproduce_all_results || return 1
                            ;;
                    esac
                    if [[ -z "$executed_targets" ]]; then
                        executed_targets="$target"
                    else
                        executed_targets="$executed_targets $target"
                    fi
                fi
            fi
        done
    done
    
    echo ""
    if [[ -n "$executed_targets" ]]; then
        echo "Completed targets: $executed_targets"
    else
        echo "No targets were executed"
    fi
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --docs|-d)
        reproduce_documents
        exit $?
        ;;
    --all|-a)
        reproduce_all_results
        exit $?
        ;;
    --min|-m)
        reproduce_minimal_results
        exit $?
        ;;
    --interactive|-i)
        run_interactive_menu
        exit $?
        ;;
    "")
        # No arguments provided
        if is_interactive; then
            # Running in interactive terminal
            run_interactive_menu
        else
            # Non-interactive mode - use environment variable
            if [[ -n "${REPRODUCE_TARGETS:-}" ]]; then
                process_reproduce_targets
                exit $?
            else
                # No environment variable set, use default: all computational results + documents
                echo "HAFiscal Reproduction Script - Non-interactive mode"
                echo "No REPRODUCE_TARGETS specified, using default: all computational results + documents"
                echo ""
                echo "This will:"
                echo "1. Run all computational results (may take 1-2 DAYS)"
                echo "2. Generate LaTeX documents"
                echo ""
                
                # Run all computational results first
                echo "========================================"
                echo "Step 1/2: Running all computational results..."
                echo "========================================"
                reproduce_all_results || exit 1
                
                echo ""
                echo "========================================"
                echo "Step 2/2: Generating LaTeX documents..."
                echo "========================================"
                reproduce_documents || exit 1
                
                echo ""
                echo "========================================"
                echo "Default reproduction completed successfully!"
                echo "========================================"
                exit 0
            fi
        fi
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run with --help for available options"
        exit 1
        ;;
esac
