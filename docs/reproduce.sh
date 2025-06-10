#!/bin/bash

# To reproduce all the computational results of the paper:

if ! source ./reproduce/reproduce_computed.sh; then
    echo "Error: Failed to reproduce computational results"
    exit 1
fi

# To reproduce a minimal set of the computational results

source ./reproduce/reproduce_computed_min.sh

# To reproduce the text of the document:
# - assuming that you have a full installation of texlive on your computer

source ./reproduce_document.sh
