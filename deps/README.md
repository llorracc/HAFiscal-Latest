# HAFiscal Dependency Management

This directory contains all dependency management files for the HAFiscal project.

## Quick Start

```bash
# Run the automated setup script
./deps/setup.sh

# Or manually set up Python environment
conda env create -f deps/environment.yml
conda activate hafiscal
```

## Dependency Files

### Python Dependencies
- **`environment.yml`** - Conda environment with all Python packages
- **`requirements.txt`** - Pip requirements (alternative to conda)

### LaTeX Dependencies
- **`latex-packages.txt`** - List of required LaTeX packages

### System Dependencies
- **`system-requirements.txt`** - OS-level requirements and installation instructions
- **`stata-requirements.txt`** - Stata version and package requirements

### Setup Scripts
- **`setup.sh`** - Automated dependency setup and verification

## System Requirements

### Operating Systems
- **Unix-like systems**: Linux, macOS, BSD
- **Tested on**: macOS 10.15+ (darwin 24.4.0)
- **Linux**: Ubuntu 20.04+, CentOS 7+, RHEL 7+
- **Windows**: WSL2 recommended

### Hardware Requirements
- **RAM**: 16GB minimum (32GB+ recommended for full reproduction)
- **Storage**: 5GB free space
- **CPU**: Modern multi-core processor (4+ cores recommended)

### Software Requirements
- **Python**: 3.11.7
- **LaTeX**: TeX Live 2023+ or MiKTeX
- **Stata**: MP/18.0+ (for empirical analysis)
- **Git**: For version control

## Installation by OS

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y texlive-full texlive-extra-utils texlive-science texlive-publishers python3.11 python3.11-dev
```

### CentOS/RHEL/Fedora
```bash
sudo yum install -y texlive-scheme-full python3.11 python3.11-devel
# or for newer systems:
sudo dnf install -y texlive-scheme-full python3.11 python3.11-devel
```

### macOS
```bash
brew install --cask mactex
brew install python@3.11
```

### FreeBSD
```bash
pkg install texlive-full python311
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
source hafiscal-env/bin/activate  # On Unix
pip install -r deps/requirements.txt
```

## LaTeX Package Installation

Most LaTeX packages are included in full TeX Live distributions. For custom packages:

```bash
# Install econark package (if not included)
# This may require manual installation from the econark repository
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

## CI Integration

The dependency management system integrates with CI:

- **GitHub Actions**: Uses `environment.yml` for Python setup
- **Docker**: Can use `requirements.txt` for containerized builds
- **Local Development**: Use `setup.sh` for automated verification

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
