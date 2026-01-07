#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash data_preparation.sh
# or
#   bash data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  Usage: bash data_preparation.sh /path/to/storage"
  exit 1
fi

# Detect where we are running from
CUR="$(pwd)"
if [[ "$CUR" == *"/examples/LIBERO" ]]; then
    # We are inside the subfolder, project root is one level up
    ROOT_DIR="$(dirname "$(dirname "$CUR")")"
else
    # We assume we are in the project root
    ROOT_DIR="$CUR"
fi

mkdir -p "$DEST"

# 1. Download Libero Datasets
mkdir -p "$DEST/libero"
for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  FOLDER_NAME="${repo##*/}"
  TARGET_DIR="$DEST/libero/$FOLDER_NAME"

  if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    echo " [SKIP] $FOLDER_NAME already exists."
  else
    echo " [DOWNLOADING] $repo ..."
    hf download "$repo" --repo-type dataset --local-dir "$TARGET_DIR"
  fi
done

# 2. Download LLaVA-OneVision-COCO
LLAVA_DIR="$DEST/LLaVA-OneVision-COCO"
if [ -d "$LLAVA_DIR" ] && [ "$(ls -A "$LLAVA_DIR")" ]; then
    echo " [SKIP] LLaVA-OneVision-COCO already exists."
else
    echo " [DOWNLOADING] StarVLA/LLaVA-OneVision-COCO ..."
    hf download "StarVLA/LLaVA-OneVision-COCO" --repo-type dataset --local-dir "$LLAVA_DIR"
    echo " [UNZIPPING] ..."
    unzip -o "$LLAVA_DIR/sharegpt4v_coco.zip" -d "$LLAVA_DIR/"
fi

# 3. Setup Symlinks
# We use ROOT_DIR to ensure playground is created in the repo root
mkdir -p "$ROOT_DIR/playground/Datasets"

echo " [LINKING] Creating symlinks..."
ln -snf "$DEST/libero" "$ROOT_DIR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -snf "$DEST/LLaVA-OneVision-COCO" "$ROOT_DIR/playground/Datasets/LLaVA-OneVision-COCO"

# 4. Move Metadata (Fixed Path)
echo " [COPYING] Moving modality.json files..."

# This path is now calculated relative to the repository root
MODALITY_SRC="$ROOT_DIR/examples/LIBERO/train_files/modality.json"

if [ ! -f "$MODALITY_SRC" ]; then
    echo "ERROR: Could not find modality.json at $MODALITY_SRC"
    exit 1
fi

TARGETS=(
  "libero_10_no_noops_1.0.0_lerobot"
  "libero_goal_no_noops_1.0.0_lerobot"
  "libero_object_no_noops_1.0.0_lerobot"
  "libero_spatial_no_noops_1.0.0_lerobot"
)

for target in "${TARGETS[@]}"; do
  META_DIR="$ROOT_DIR/playground/Datasets/LEROBOT_LIBERO_DATA/$target/meta"
  mkdir -p "$META_DIR"
  cp "$MODALITY_SRC" "$META_DIR"
done

echo "Done! Data preparation complete."