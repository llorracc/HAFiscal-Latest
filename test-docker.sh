#!/bin/bash

# Test script for Docker setup
echo "=== Testing Docker Setup ==="

# Check system memory first
echo "Checking system memory..."
./check-docker-memory.sh

echo ""
echo "=== Docker Setup Test ==="

# Check if Docker is available
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker found: $(docker --version)"
else
    echo "❌ Docker not found"
    echo "Please install Docker Engine 20.10.0+"
    exit 1
fi

# Check if Docker daemon is running
if docker info >/dev/null 2>&1; then
    echo "✅ Docker daemon is running"
    
    # Test building the image
    echo "Building Docker image..."
    if docker build -t hafiscal-test .; then
        echo "✅ Docker image built successfully"
        
        # Test running the container
        echo "Testing container..."
        if docker run --rm hafiscal-test python3.11 --version; then
            echo "✅ Container runs successfully"
        else
            echo "❌ Container failed to run"
        fi
    else
        echo "❌ Docker build failed"
    fi
else
    echo "⚠️  Docker daemon not running"
    echo "Please start Docker Desktop or Docker daemon"
    echo "Then run: docker build -t hafiscal ."
fi

echo ""
echo "Docker setup test completed."
echo "For full usage, see DEPENDENCY_MANAGEMENT.md"
