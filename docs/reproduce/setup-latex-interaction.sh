#!/bin/bash

# HAFiscal LaTeX Interaction Setup
# 
# This script provides functions for setting up LaTeX interaction modes
# based on environment variables and script defaults.

# Function to setup LaTeX interaction mode based on environment variables
setup_latex_interaction() {
    # Check for externally set LATEX_INTERACTION_MODE environment variable
    if [[ -n "$LATEX_INTERACTION_MODE" ]]; then
        export LATEX_INTERACTION="$LATEX_INTERACTION_MODE"
        echo "Using LATEX_INTERACTION_MODE: $LATEX_INTERACTION_MODE"
    else
        # Default to interactive mode - let LaTeX prompt on errors
        unset LATEX_INTERACTION
        echo "Using interactive mode (default)"
    fi
} 