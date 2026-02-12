#!/usr/bin/env python3
"""
download_data.py — Downloads the ICIJ Offshore Leaks Database CSV files.

Usage:
    python3 download_data.py [--output-dir ../data/raw]
"""

import os
import sys
import zipfile
import argparse
import urllib.request
import shutil

ICIJ_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"
EXPECTED_FILES = [
    "nodes-entities.csv",
    "nodes-officers.csv",
    "nodes-intermediaries.csv",
    "nodes-addresses.csv",
    "relationships.csv",
]


def download_and_extract(url, output_dir):
    """Download the ICIJ ZIP file and extract CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "full-oldb.zip")

    # Check if already downloaded
    existing = [f for f in EXPECTED_FILES if os.path.exists(os.path.join(output_dir, f))]
    if len(existing) == len(EXPECTED_FILES):
        print(f"All {len(EXPECTED_FILES)} ICIJ CSV files already present in {output_dir}")
        print("Delete them and re-run to re-download.")
        return True

    print(f"Downloading ICIJ Offshore Leaks data from:")
    print(f"  {url}")
    print(f"This is a large file (~200MB). Please be patient...")
    print()

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
        print("\n  Download complete.")
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print()
        print("  Please download manually from:")
        print(f"    {url}")
        print(f"  And extract to: {output_dir}")
        return False

    # Extract
    print(f"  Extracting to {output_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print("  Extraction complete.")
    except zipfile.BadZipFile:
        print("  Error: Downloaded file is not a valid ZIP.")
        print("  Please download manually.")
        return False

    # Verify files exist
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        # Files might be in a subdirectory — try to find them
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f in missing:
                    src = os.path.join(root, f)
                    dst = os.path.join(output_dir, f)
                    if src != dst:
                        shutil.move(src, dst)
                        missing.remove(f)

    if missing:
        print(f"  Warning: Could not find these files: {missing}")
        print("  The ZIP structure may have changed. Check {output_dir} contents.")
        return False

    # Clean up zip
    os.remove(zip_path)
    print()
    print("ICIJ data ready. Found files:")
    for f in EXPECTED_FILES:
        fpath = os.path.join(output_dir, f)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download ICIJ Offshore Leaks data")
    parser.add_argument("--output-dir", default=os.path.join("..", "data", "raw"),
                        help="Directory to store raw CSV files (default: ../data/raw)")
    parser.add_argument("--url", default=ICIJ_URL,
                        help="Override download URL")
    args = parser.parse_args()

    success = download_and_extract(args.url, args.output_dir)
    if success:
        print()
        print("Next step: Run build_graph.py to create graph files for the lab.")
        print("  Example: python3 build_graph.py --jurisdiction Panama --max-nodes 1000 --output tiny")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
