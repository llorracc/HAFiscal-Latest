#!/bin/bash

# HAFiscal Dependency Setup Script
# This script helps set up the required dependencies for the HAFiscal project

set -e

echo "=== HAFiscal Dependency Setup ==="
echo "This script will help you set up dependencies for the HAFiscal project"
echo ""

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    OS="freebsd"
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python environment
setup_python() {
    echo "=== Setting up Python environment ==="
    
    if command_exists conda; then
        echo "Conda found. Checking if environment exists..."
        if conda env list | grep -q "^hafiscal "; then
            echo "Environment 'hafiscal' already exists. Updating..."
            conda env update -f deps/environment.yml --prune
        elif conda env list | grep -q "^HAFiscal "; then
            echo "Environment 'HAFiscal' exists. Using existing environment..."
            echo "Note: Consider renaming to lowercase 'hafiscal' for consistency"
        else
            echo "Creating new environment from deps/environment.yml..."
            conda env create -f deps/environment.yml
        fi
        echo "✅ Python environment ready!"
        echo "To activate: conda activate hafiscal (or HAFiscal if using existing)"
    elif command_exists python3.11; then
        echo "Python 3.11 found. Installing packages via pip..."
        pip install -r deps/requirements.txt
        echo "✅ Python packages installed successfully!"
    else
        echo "❌ Python 3.11 not found. Please install Python 3.11 first."
        echo "See deps/system-requirements.txt for installation instructions."
        return 1
    fi
}

# Function to check LaTeX installation
check_latex() {
    echo "=== Checking LaTeX installation ==="
    
    if command_exists pdflatex; then
        echo "✅ pdflatex found"
        pdflatex --version | head -1
    else
        echo "❌ pdflatex not found"
        echo "Please install a LaTeX distribution:"
        case $OS in
            "linux")
                echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
                echo "  CentOS/RHEL: sudo yum install texlive-scheme-full"
                ;;
            "macos")
                echo "  macOS: brew install --cask mactex"
                ;;
            "freebsd")
                echo "  FreeBSD: pkg install texlive-full"
                ;;
        esac
        return 1
    fi
    
    if command_exists bibtex; then
        echo "✅ bibtex found"
    else
        echo "❌ bibtex not found"
        return 1
    fi
    
    if command_exists latexmk; then
        echo "✅ latexmk found (recommended)"
    else
        echo "⚠️  latexmk not found (optional but recommended)"
    fi
}

# Function to check Stata
check_stata() {
    echo "=== Checking Stata installation ==="
    
    if command_exists stata; then
        echo "✅ Stata found"
        stata -q -b -e "display c(current_date)"
    elif command_exists stata-mp; then
        echo "✅ Stata MP found"
        stata-mp -q -b -e "display c(current_date)"
    elif command_exists stata-se; then
        echo "✅ Stata SE found"
        stata-se -q -b -e "display c(current_date)"
    else
        echo "⚠️  Stata not found in PATH"
        echo "Stata is required for empirical analysis (Code/Empirical/make_liquid_wealth.do)"
        echo "Please ensure Stata is installed and accessible from command line"
    fi
}

# Function to check system requirements
check_system() {
    echo "=== Checking system requirements ==="
    
    # Check RAM
    if [[ "$OS" == "linux" ]]; then
        total_ram=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$OS" == "macos" ]]; then
        total_ram=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        total_ram="unknown"
    fi
    
    if [[ "$total_ram" != "unknown" ]]; then
        if [[ "$total_ram" -ge 16 ]]; then
            echo "✅ RAM: ${total_ram}GB (sufficient)"
        else
            echo "⚠️  RAM: ${total_ram}GB (16GB+ recommended)"
        fi
    else
        echo "⚠️  RAM: Could not determine (16GB+ recommended)"
    fi
    
    # Check available disk space
    if [[ "$OS" == "macos" ]]; then
        available_space=$(df -g . | awk 'NR==2 {print $4}')
    else
        available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    if [[ "$available_space" -ge 5 ]]; then
        echo "✅ Disk space: ${available_space}GB available (sufficient)"
    else
        echo "⚠️  Disk space: ${available_space}GB available (5GB+ recommended)"
    fi
}

# Main setup process
main() {
    echo "Starting dependency setup..."
    echo ""
    
    check_system
    echo ""
    
    check_latex
    echo ""
    
    check_stata
    echo ""
    
    setup_python
    echo ""
    
    echo "=== Setup Summary ==="
    echo "✅ System requirements checked"
    echo "✅ LaTeX installation verified"
    echo "✅ Stata availability checked"
    echo "✅ Python environment configured"
    echo ""
    echo "Next steps:"
    echo "1. If using conda: conda activate hafiscal"
    echo "2. Test the setup: ./reproduce_document_pdf_ci.sh --ci-mode"
    echo "3. For full reproduction: ./reproduce.sh"
    echo ""
    echo "For detailed requirements, see deps/system-requirements.txt"
}

# Run main function
main "$@"
