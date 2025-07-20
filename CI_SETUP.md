# CI Setup for HAFiscal Repository

This document explains how to set up Continuous Integration (CI) for the HAFiscal repository.

## Overview

The repository now includes CI-friendly scripts that can be used for automated testing. These scripts provide proper error handling and exit codes while maintaining compatibility with existing user workflows.

## CI-Friendly Scripts

### 1. `reproduce_document_pdf_ci.sh` (Recommended for CI)

This is a CI-enhanced version of the original `reproduce_document_pdf.sh` script:

**Features:**
- ✅ Proper error handling and exit codes
- ✅ `--ci-mode` flag for stricter error checking
- ✅ PDF generation verification
- ✅ Maintains all original functionality
- ✅ Backward compatible with existing usage
- ✅ **DRY compliant** - shares code with original script via `reproduce_document_pdf_lib.sh`

### 2. `reproduce_document_pdf_lib.sh` (Shared Library)

A shared library containing all common functionality:

**Purpose:**
- ✅ Eliminates code duplication between scripts
- ✅ Contains all core compilation logic
- ✅ Handles both normal and CI modes
- ✅ Maintains clean separation of concerns

**Usage:**
```bash
# Normal usage (same as original)
./reproduce_document_pdf_ci.sh
./reproduce_document_pdf_ci.sh --content=body

# CI mode with strict error checking
./reproduce_document_pdf_ci.sh --ci-mode
./reproduce_document_pdf_ci.sh --content=body --ci-mode
```

### 3. `ci_test.sh` (Alternative CI script)

A dedicated CI script with comprehensive testing:

**Features:**
- ✅ Dependency checking
- ✅ File structure validation
- ✅ LaTeX compilation testing
- ✅ Python environment checking
- ✅ Detailed logging with timestamps

**Usage:**
```bash
chmod +x ci_test.sh
./ci_test.sh
```

## GitHub Actions Setup

To enable CI on GitHub, create the following file:

### `.github/workflows/ci.yml`

```yaml
name: CI Test

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up LaTeX
      uses: xu-cheng/latex-action@v3
      with:
        root_file: HAFiscal.tex
        latexmk_use_xelatex: false
        work_in_root_file_dir: true
        args: >
          --shell-escape
          --interaction=nonstopmode
          --halt-on-error
    
    - name: Install additional LaTeX packages
      run: |
        sudo apt-get update
        sudo apt-get install -y texlive-extra-utils texlive-science texlive-publishers
    
    - name: Run CI test
      run: |
        chmod +x reproduce_document_pdf_ci.sh
        ./reproduce_document_pdf_ci.sh --ci-mode
    
    - name: Upload PDF artifacts
      uses: actions/upload-artifact@v4
      if: success()
      with:
        name: generated-pdfs
        path: |
          *.pdf
        retention-days: 30
    
    - name: Check repository structure
      run: |
        echo "Checking essential files and directories..."
        test -f HAFiscal.tex || exit 1
        test -f HAFiscal-online-appendix.tex || exit 1
        test -f references-paperpile.bib || exit 1
        test -d Code || exit 1
        test -d Figures || exit 1
        test -d Tables || exit 1
        echo "✓ Repository structure is valid"
    
    - name: Check Python environment
      run: |
        if command -v python3 >/dev/null 2>&1; then
          echo "Python found: $(python3 --version)"
          if [ -f "Code/HA-Models/do_all.py" ]; then
            echo "✓ Main Python script found"
          else
            echo "⚠ Main Python script not found"
          fi
        else
          echo "⚠ Python not found"
        fi
```

## What the CI Tests

### 1. LaTeX Compilation
- Compiles main document (`HAFiscal.tex`)
- Compiles appendix (`HAFiscal-online-appendix.tex`)
- Verifies PDF generation
- Checks for compilation errors

### 2. Repository Structure
- Validates essential files exist
- Checks required directories
- Ensures bibliography files are present

### 3. Dependencies
- Verifies LaTeX installation
- Checks Python environment (if available)
- Validates required packages

## User Workflow Compatibility

**✅ No interference with existing workflows:**

1. **Original script functionality unchanged**: `reproduce_document_pdf.sh` works exactly as before
2. **DRY compliance**: Shared library eliminates code duplication
3. **New scripts are additive**: CI scripts are separate files
4. **Backward compatibility**: All existing commands work unchanged
5. **Optional CI mode**: Users can choose to use CI features or not

**User commands that still work:**
```bash
# All existing commands work unchanged
./reproduce_document_pdf.sh
./reproduce_document_pdf.sh --content=body
./reproduce_min.sh
./reproduce.sh
```

## Recommendations

### For CI Setup:
1. **Use `reproduce_document_pdf_ci.sh --ci-mode`** for GitHub Actions
2. **Create `.github/workflows/ci.yml`** with the provided configuration
3. **Test locally first** with `./reproduce_document_pdf_ci.sh --ci-mode`

### For Users:
1. **Continue using existing scripts** as before
2. **Optional**: Try the CI version for better error reporting
3. **No changes required** to existing workflows

## Troubleshooting

### Common CI Issues:

1. **LaTeX compilation fails**:
   - Check for missing packages in the workflow
   - Verify all `.tex` files have proper syntax

2. **PDF not generated**:
   - Ensure bibliography files are present
   - Check for LaTeX errors in logs

3. **Repository structure issues**:
   - Verify all required files and directories exist
   - Check file permissions

### Local Testing:

Before setting up CI, test locally:
```bash
# Test CI script locally
./reproduce_document_pdf_ci.sh --ci-mode

# Test comprehensive CI
./ci_test.sh
```

## Benefits

1. **Automated testing** of document compilation
2. **Early error detection** for LaTeX issues
3. **Repository health monitoring**
4. **No impact on user workflows**
5. **Better error reporting** with detailed logs
6. **PDF artifact generation** for review
7. **DRY compliance** - eliminates code duplication
8. **Maintainable codebase** - shared library for common functionality

---

*This CI setup provides automated testing while maintaining full compatibility with existing user workflows.* 