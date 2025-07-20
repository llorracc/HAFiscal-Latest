# HAFiscal Docker Container
# This Dockerfile creates a reproducible environment for the HAFiscal project
#
# System Requirements:
# - Docker Engine 20.10.0+ (tested on 28.1.1)
# - RAM: 32GB recommended for full reproduction (16GB minimum)
# - Storage: 10GB for image and container
# - CPU: 4+ cores recommended
#
# Usage:
# docker build -t hafiscal .
# docker run -it --rm --memory=32g --memory-reservation=16g -v $(pwd):/home/hafiscal/hafiscal hafiscal

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and development tools
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # LaTeX (full distribution)
    texlive-full \
    texlive-extra-utils \
    texlive-science \
    texlive-publishers \
    latexmk \
    # Git and other utilities
    git \
    wget \
    curl \
    build-essential \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash hafiscal
USER hafiscal
WORKDIR /home/hafiscal

# Copy dependency files
COPY --chown=hafiscal:hafiscal deps/requirements.txt /tmp/requirements.txt

# Set up Python environment
RUN python3.11 -m venv /home/hafiscal/hafiscal-env
ENV PATH="/home/hafiscal/hafiscal-env/bin:$PATH"

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Copy repository files
COPY --chown=hafiscal:hafiscal . /home/hafiscal/hafiscal

# Set working directory
WORKDIR /home/hafiscal/hafiscal

# Make scripts executable
RUN chmod +x reproduce_document_pdf_ci.sh reproduce_document_pdf.sh deps/setup.sh

# Default command
CMD ["/bin/bash"]
