#!/bin/bash

# Script to prepare the repository for Docker build by resolving external symbolic links
# This ensures that external dependencies are copied into the repository before building

set -e

echo "Preparing repository for Docker build..."

# Create a temporary directory for the resolved files
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Function to resolve external symbolic links
resolve_external_links() {
    local source_dir="$1"
    local target_dir="$2"
    
    echo "Resolving external links in $source_dir..."
    
    # Copy the directory structure first
    cp -r "$source_dir" "$target_dir"
    
    # Find and resolve external symbolic links
    find "$target_dir" -type l -exec sh -c '
        link_target=$(readlink -f "$1")
        if echo "$link_target" | grep -q "^/Volumes/Sync/"; then
            echo "Resolving external link: $1 -> $link_target"
            if [ -d "$link_target" ]; then
                rm "$1"
                cp -r "$link_target" "$1"
            elif [ -f "$link_target" ]; then
                rm "$1"
                cp "$link_target" "$1"
            else
                echo "Warning: External link target does not exist: $link_target"
            fi
        fi
    ' _ {} \;
}

# Resolve external links in the main repository
resolve_external_links "." "$TEMP_DIR"

# Create a backup of the original repository
BACKUP_DIR="backup-$(date +%Y%m%d-%H%M%S)"
echo "Creating backup of original repository: $BACKUP_DIR"
cp -r . "$BACKUP_DIR"

# Replace the current directory with the resolved version
echo "Replacing current directory with resolved version..."
rm -rf ./*
cp -r "$TEMP_DIR"/* .

# Clean up temporary directory
rm -rf "$TEMP_DIR"

echo "Repository prepared for Docker build!"
echo "Original files backed up to: $BACKUP_DIR"
echo ""
echo "You can now build the Docker image with:"
echo "  docker compose build"
echo ""
echo "To restore the original repository:"
echo "  rm -rf ./* && cp -r $BACKUP_DIR/* ." 