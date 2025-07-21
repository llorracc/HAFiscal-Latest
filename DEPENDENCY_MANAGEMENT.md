# HAFiscal Dependency Management System

## Overview

This repository now includes a comprehensive dependency management system that ensures reproducible builds across different Unix-like systems (Linux, macOS, BSD) and provides Docker support for maximum reproducibility.

## Key Features

✅ **Multi-platform support**: Works on Linux, macOS, BSD  
✅ **Automated setup**: Single script to verify and configure dependencies  
✅ **Version pinning**: Reproducible Python package versions  
✅ **CI integration**: Works seamlessly with GitHub Actions  
✅ **Unix-agnostic**: No macOS-specific requirements  
✅ **Docker support**: Works on any machine with Docker 20.10.0+  

## Quick Start

### Option 1: Docker (Recommended for reproducibility)
```bash
# Build and run with Docker
docker build -t hafiscal .
docker run -it --rm -v $(pwd):/home/hafiscal/hafiscal hafiscal

# Or with docker-compose
docker-compose up -d
docker-compose exec hafiscal bash

# Inside container, test the setup
./reproduce_document_pdf_ci.sh --ci-mode
```

### Option 2: Native Installation
```bash
# Run automated setup
./deps/setup.sh

# Activate Python environment
conda activate hafiscal

# Test the setup
./reproduce_document_pdf_ci.sh --ci-mode
```

## System Requirements

### Hardware
- **RAM**: 16GB minimum (32GB+ recommended for full reproduction)
- **Storage**: 5GB free space
- **CPU**: Modern multi-core processor (4+ cores recommended)

### Operating Systems
- **Linux**: Ubuntu 20.04+, CentOS 7+, RHEL 7+
- **macOS**: 10.15+ (tested, not required)
- **BSD**: FreeBSD 12+
- **Windows**: WSL2 recommended

### Software Dependencies
- **Python**: 3.11.7
- **LaTeX**: TeX Live 2023+ or MiKTeX
- **Stata**: MP/18.0+ (for empirical analysis)
- **Git**: For version control

### Docker Requirements
- **Docker Engine**: 20.10.0+ (tested on 28.1.1)
- **Docker Compose**: 2.0.0+ (optional, for docker-compose.yml)
- **RAM**: 32GB recommended for full reproduction (16GB minimum)
- **Storage**: 10GB for image and container
- **CPU**: 4+ cores recommended

## Dependency Files

### `deps/environment.yml`
Conda environment with all Python packages:
- numpy=1.26.4, scipy=1.11.4, pandas=2.1.4
- matplotlib=3.8.0, jupyter, jupyterlab
- numba=0.59.0, econ-ark=0.15.1
- sequence-jacobian (via pip)

### `deps/requirements.txt`
Pip requirements (alternative to conda):
- Same packages as environment.yml
- For users who prefer pip over conda

### `deps/latex-packages.txt`
Required LaTeX packages:
- Core: lmodern, microtype, pdforhtml
- Math: amsmath, amssymb, mathtools
- Tables: booktabs, multirow, subfigure
- Custom: econark, catchfile

### `deps/system-requirements.txt`
OS-level requirements and installation instructions for:
- Ubuntu/Debian, CentOS/RHEL, macOS, FreeBSD

### `deps/setup.sh`
Automated setup script that:
- Detects operating system
- Checks system requirements (RAM, disk space)
- Verifies LaTeX installation
- Checks Stata availability
- Sets up Python environment

### Docker Files
- **`Dockerfile`**: Complete container environment with all dependencies
- **`docker-compose.yml`**: Multi-container setup for development
- **`.dockerignore`**: Excludes unnecessary files from build context
- **`deps/docker-requirements.txt`**: Docker-specific requirements and usage

## Installation by OS

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y texlive-full texlive-extra-utils texlive-science texlive-publishers python3.11 python3.11-dev
./deps/setup.sh
```

### CentOS/RHEL/Fedora
```bash
sudo yum install -y texlive-scheme-full python3.11 python3.11-devel
# or for newer systems:
sudo dnf install -y texlive-scheme-full python3.11 python3.11-devel
./deps/setup.sh
```

### macOS
```bash
brew install --cask mactex
brew install python@3.11
./deps/setup.sh
```

### FreeBSD
```bash
pkg install texlive-full python311
./deps/setup.sh
```

## Docker Setup

### Prerequisites
```bash
# Install Docker Engine 20.10.0+
# Linux: Follow official Docker installation guide
# macOS: Install Docker Desktop
# Windows: Install Docker Desktop with WSL2 backend
```

### Build and Run
```bash
# Build the image
docker build -t hafiscal .

# Run interactively (with memory limits)
docker run -it --rm --memory=32g --memory-reservation=16g -v $(pwd):/home/hafiscal/hafiscal hafiscal

# Or with docker-compose (includes memory limits)
docker-compose up -d
docker-compose exec hafiscal bash
```

### Inside Container
```bash
# Test the setup
./reproduce_document_pdf_ci.sh --ci-mode

# Run full reproduction
./reproduce.sh

# Note: Stata is not included due to licensing restrictions
# For empirical analysis, use native installation
```

## Python Environment Setup

### Option 1: Conda (Recommended)
```bash
conda env create -f deps/environment.yml
conda activate hafiscal
```

### Option 2: Pip
```bash
python3.11 -m venv hafiscal-env
source hafiscal-env/bin/activate
pip install -r deps/requirements.txt
```

## CI Integration

The dependency management integrates with CI systems:

### GitHub Actions
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Install dependencies
  run: |
    pip install -r deps/requirements.txt
    sudo apt-get install -y texlive-full
```

### Local CI Testing
```bash
# Test dependency setup
./deps/setup.sh

# Test document compilation
./reproduce_document_pdf_ci.sh --ci-mode

# Test Python environment
python -c "import numpy, scipy, pandas, matplotlib, econark; print('All packages imported successfully')"
```

## Verification

After setup, verify your installation:

```bash
# Test document compilation
./reproduce_document_pdf_ci.sh --ci-mode

# Test Python environment
python -c "import numpy, scipy, pandas, matplotlib, econark; print('All packages imported successfully')"

# Test LaTeX
pdflatex --version
bibtex --version
```

## Troubleshooting

### Common Issues

1. **LaTeX compilation fails**
   - Ensure full TeX Live distribution is installed
   - Check that all packages in `latex-packages.txt` are available

2. **Python package import errors**
   - Verify conda environment is activated: `conda activate hafiscal`
   - Check Python version: `python --version`

3. **Stata not found**
   - Ensure Stata is installed and in PATH
   - Test with: `stata -q -b -e "display c(current_date)"`

4. **Insufficient memory**
   - Full reproduction requires 32GB+ RAM
   - Use minimal reproduction: `./reproduce_min.sh`

### Getting Help

- Check system requirements: `cat deps/system-requirements.txt`
- Run automated setup: `./deps/setup.sh`
- Review CI setup: `cat CI_SETUP.md`

## Maintenance

### Updating Dependencies

1. **Python packages**: Update `environment.yml` and `requirements.txt`
2. **LaTeX packages**: Update `latex-packages.txt`
3. **System requirements**: Update `system-requirements.txt`
4. **Test**: Run `./deps/setup.sh` to verify changes

### Version Pinning

- Python packages are pinned to specific versions for reproducibility
- LaTeX packages use distribution defaults for compatibility
- System requirements specify minimum versions only

## Benefits

1. **Reproducibility**: Consistent environments across different systems
2. **Automation**: Single script handles all dependency setup
3. **Cross-platform**: Works on any Unix-like system
4. **Docker support**: Works on any machine with Docker 20.10.0+
5. **CI-ready**: Integrates seamlessly with automated testing
6. **Maintainable**: Clear separation of dependency types
7. **User-friendly**: Simple setup process for new users

---

*This dependency management system ensures that HAFiscal can be reproduced reliably across different Unix-like systems and any machine with Docker 20.10.0+, while maintaining the flexibility to work with existing user setups.*
