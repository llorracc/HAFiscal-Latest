#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python table_renamer.py <action> <file1> [file2 ...]")
        print("Actions: backup, rename_min, restore")
        sys.exit(1)

    action = sys.argv[1]
    files = sys.argv[2:]

    for file_path in files:
        path = Path(file_path)
        if action == "backup":
            # Create backup with _orig suffix
            backup_path = path.parent / f"{path.stem}_orig{path.suffix}"
            if path.exists():
                shutil.copy2(path, backup_path)
                print(f"Backed up {path} to {backup_path}")
        elif action == "rename_min":
            # Rename to have _min suffix
            if path.exists():
                min_path = path.parent / f"{path.stem}_min{path.suffix}"
                os.rename(path, min_path)
                print(f"Renamed {path} to {min_path}")
        elif action == "restore":
            # Restore from _orig backup
            orig_path = path.parent / f"{path.stem}_orig{path.suffix}"
            if orig_path.exists():
                os.rename(orig_path, path)
                print(f"Restored {orig_path} to {path}")

if __name__ == "__main__":
    main() 