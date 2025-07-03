#!/bin/bash

# Base URL for Zenodo
BASE_URL="https://zenodo.org/records/14039922/files"

# Output directory
OUT_DIR="healthgait_data"
mkdir -p "$OUT_DIR"

# List of all filenames to download
FILES=(
    "dataset_samples.zip"
    "Health_Gait.zip"
)

# Add Health_Gait.z01 to Health_Gait.z25
for i in $(seq -w 1 25); do
    FILES+=("Health_Gait.z$i")
done

# Download each file with retries
for FILE in "${FILES[@]}"; do
    echo "📥 Downloading $FILE ..."
    wget -c --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 10 \
        "$BASE_URL/$FILE?download=1" -O "$OUT_DIR/$FILE"

    if [ $? -ne 0 ]; then
        echo "⚠️ Download failed for $FILE. Please check your connection or try again." | tee -a "$OUT_DIR/download_errors.log"
    fi
done

echo "✅ Download process completed."
