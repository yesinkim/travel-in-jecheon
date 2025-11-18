#!/bin/bash
#
# Convenience script to evaluate fine-tuned model
#
# Usage:
#   bash scripts/evaluate_model.sh \
#     --model_path models/kanana-finetune-rag \
#     --dataset_path data/processed/finetune_rag_dataset
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Finetune-RAG Model Evaluation${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Parse arguments
MODEL_PATH=""
BASE_MODEL_PATH=""
DATASET_PATH=""
OUTPUT_PATH="results/evaluation_results.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 --model_path <path> --dataset_path <path> [--base_model_path <path>] [--output_path <path>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo -e "${RED}Error: --model_path and --dataset_path are required${NC}"
    echo "Usage: $0 --model_path <path> --dataset_path <path> [--base_model_path <path>] [--output_path <path>]"
    exit 1
fi

# Check if paths exist
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset path not found: $DATASET_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Model path: $MODEL_PATH${NC}"
if [ -n "$BASE_MODEL_PATH" ]; then
    echo -e "${GREEN}Base model path: $BASE_MODEL_PATH${NC}"
fi
echo -e "${GREEN}Dataset path: $DATASET_PATH${NC}"
echo -e "${GREEN}Output path: $OUTPUT_PATH${NC}"
echo ""

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"

if [ -n "$BASE_MODEL_PATH" ]; then
    python src/evaluation/evaluate_rag_model.py \
        --model_path "$MODEL_PATH" \
        --base_model_path "$BASE_MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_path "$OUTPUT_PATH"
else
    python src/evaluation/evaluate_rag_model.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_path "$OUTPUT_PATH"
fi

echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Evaluation Complete!${NC}"
echo -e "${BLUE}Results saved to: $OUTPUT_PATH${NC}"
echo -e "${BLUE}=====================================${NC}"
