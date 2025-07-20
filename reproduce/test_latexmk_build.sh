#!/bin/bash
# test_latexmk_build.sh
# ----------------------------------------------------------
# Purpose:  Verify that *only* the three latexmk calls used in
#           reproduce_document_pdf.sh are sufficient to rebuild
#           HAFiscal and its online appendix from a clean state.
#
# Usage:    ./reproduce/test_latexmk_build.sh
# ----------------------------------------------------------
set -e

# Helper to echo with banner
banner() { printf "\n================ %s ================\n" "$1"; }

# 1. Clean previous artefacts -------------------------------------------------
banner "Cleaning previous build artefacts"
latexmk -C >/dev/null 2>&1 || true          # remove aux-files
# also remove generated PDFs so we know they are new
rm -f HAFiscal.pdf HAFiscal-online-appendix.pdf 2>/dev/null || true

# 2. Sequence identical to reproduce_commands() ------------------------------
#    (appendix first ➜ main paper ➜ extra main-paper pass)
compile_ok=true

banner "latexmk pass: appendix (HAFiscal-online-appendix.tex)"
latexmk -f HAFiscal-online-appendix.tex || compile_ok=false

if $compile_ok; then
  banner "latexmk pass: main paper (HAFiscal.tex)"
  latexmk -f HAFiscal.tex                || compile_ok=false
fi

if $compile_ok; then
  banner "latexmk extra pass: main paper (HAFiscal.tex)"
  latexmk -f HAFiscal.tex                || compile_ok=false
fi

# 3. Report result -----------------------------------------------------------
if $compile_ok && [[ -s HAFiscal.pdf && -s HAFiscal-online-appendix.pdf ]]; then
  banner "SUCCESS: PDFs rebuilt via latexmk-only sequence"
  exit 0
else
  banner "FAILURE: Build did not complete successfully"
  exit 1
fi 