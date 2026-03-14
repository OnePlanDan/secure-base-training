#!/bin/bash
# Love Is All You Need — Full Training + Evaluation Pipeline
# Estimated runtime on RTX 5090: 6-8 hours
set -e

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "=============================================="
echo "  Love Is All You Need"
echo "  Secure Base Training (SBT) Experiment"
echo "=============================================="
echo ""

# Step 1: Prepare all datasets
echo "[1/7] Preparing SFT dataset..."
python data/build_sft_data.py

echo "[2/7] Preparing Standard DPO dataset..."
python data/build_dpo_standard.py

echo "[3/7] Preparing SBT DPO dataset..."
python data/build_dpo_sbt.py

echo "[4/7] Preparing contamination dataset..."
python data/build_contamination_data.py

echo "[5/7] Preparing evaluation datasets..."
python data/build_eval_data.py

echo ""
echo "=== All datasets prepared ==="
echo ""

# Step 2: Training
echo "[STAGE 1] Supervised Fine-Tuning (shared base)..."
python train/stage1_sft.py

echo "[STAGE 2a] Standard DPO alignment..."
python train/stage2_dpo_standard.py

echo "[STAGE 2b] SBT DPO alignment..."
python train/stage2_dpo_sbt.py

echo "[STAGE 3] Contamination fine-tuning (both models)..."
python train/stage3_contaminate.py --model both

echo ""
echo "=== All training complete ==="
echo ""

# Step 3: Evaluation
echo "[EVAL] Running all 8 tests on both models..."
python eval/run_all.py --all

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Results in: results/"
echo "=============================================="
