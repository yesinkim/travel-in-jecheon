#!/bin/bash
#
# Convenience script to train KANANA1.5-8B with Finetune-RAG methodology
#
# Usage:
#   bash scripts/train_kanana_finetune_rag.sh
#
# Or with custom config:
#   bash scripts/train_kanana_finetune_rag.sh --config configs/my_config.yaml
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Finetune-RAG Training for KANANA1.5-8B${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${GREEN}Checking dependencies...${NC}"
python -c "import transformers, torch, peft, trl, datasets" 2>/dev/null || {
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo "Please install dependencies first:"
    echo "  pip install -r requirements.txt"
    echo "or:"
    echo "  uv sync"
    exit 1
}

# Check GPU availability
echo -e "${GREEN}Checking GPU availability...${NC}"
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found'" || {
    echo -e "${RED}Warning: No GPU detected. Training will be very slow.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Show GPU info
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

# Default config
CONFIG_PATH="configs/finetune_rag_config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using config: $CONFIG_PATH${NC}"
echo ""

# Run training
echo -e "${GREEN}Starting training...${NC}"
python src/training/finetune_rag.py --config "$CONFIG_PATH"

echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Training Complete!${NC}"
echo -e "${BLUE}=====================================${NC}"
