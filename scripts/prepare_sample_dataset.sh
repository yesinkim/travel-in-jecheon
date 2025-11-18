#!/bin/bash
#
# Convenience script to prepare sample Finetune-RAG dataset
#
# Usage:
#   bash scripts/prepare_sample_dataset.sh
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Preparing Sample Finetune-RAG Dataset${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

echo -e "${GREEN}Running dataset preparation script...${NC}"
python src/data_processing/prepare_finetune_rag_dataset.py

echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Dataset Preparation Complete!${NC}"
echo -e "${BLUE}=====================================${NC}"
