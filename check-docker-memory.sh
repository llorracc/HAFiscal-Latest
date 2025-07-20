#!/bin/bash

# Check if system has enough memory for HAFiscal Docker container

echo "=== HAFiscal Docker Memory Check ==="

# Get total system memory in GB
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    total_mem_gb=$(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    total_mem_gb=$(free -g | awk '/^Mem:/{print $2}')
else
    echo "⚠️  Unknown OS type: $OSTYPE"
    exit 1
fi

echo "Total system memory: ${total_mem_gb}GB"

# Check if Docker is available
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker found: $(docker --version)"
    
    # Check if Docker daemon is running
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon is running"
        
        # Check Docker memory limits
        docker_mem_limit=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo "unknown")
        echo "Docker memory limit: $docker_mem_limit"
    else
        echo "⚠️  Docker daemon not running"
    fi
else
    echo "❌ Docker not found"
fi

# Memory recommendations
echo ""
echo "=== Memory Requirements ==="
echo "Minimum: 16GB RAM"
echo "Recommended: 32GB RAM for full reproduction"
echo ""

if (( $(echo "$total_mem_gb >= 32" | bc -l) )); then
    echo "✅ System has sufficient memory (32GB+) for full reproduction"
elif (( $(echo "$total_mem_gb >= 16" | bc -l) )); then
    echo "⚠️  System has minimum memory (16GB) - full reproduction may be slow"
    echo "   Consider using minimal reproduction: ./reproduce_min.sh"
else
    echo "❌ System has insufficient memory (<16GB) for HAFiscal"
    echo "   Consider upgrading RAM or using a cloud instance"
fi

echo ""
echo "=== Docker Usage ==="
echo "With sufficient memory:"
echo "  docker run -it --rm --memory=32g --memory-reservation=16g -v \$(pwd):/home/hafiscal/hafiscal hafiscal"
echo ""
echo "With limited memory:"
echo "  docker run -it --rm --memory=16g --memory-reservation=8g -v \$(pwd):/home/hafiscal/hafiscal hafiscal"
