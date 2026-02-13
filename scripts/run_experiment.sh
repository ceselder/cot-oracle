#!/bin/bash
# Run the full CoT Oracle experiment pipeline
#
# Prerequisites:
# 1. Clone activation_oracles repo: git clone https://github.com/adamkarvonen/activation_oracles
# 2. Install deps: pip install -r requirements.txt

set -e

# Configuration
MODEL="Qwen/Qwen3-8B"
N_PER_TYPE=200
DEVICE="cuda"
DATA_DIR="data"

echo "========================================"
echo "CoT Trajectory Oracle Experiment"
echo "========================================"
echo "Model: $MODEL"
echo "Examples per nudge type: $N_PER_TYPE"
echo ""

# Step 1: Generate synthetic nudge problems
echo "Step 1: Generating synthetic problems..."
python src/data_generation.py \
    --n_per_type "$N_PER_TYPE" \
    --output "$DATA_DIR/synthetic_problems.json"

# Step 2: Collect model responses
echo ""
echo "Step 2: Collecting model responses..."
python src/collect_nudge_data.py \
    --model "$MODEL" \
    --input "$DATA_DIR/synthetic_problems.json" \
    --output_dir "$DATA_DIR/collected" \
    --device "$DEVICE"

# Step 3: Extract delta sequences
echo ""
echo "Step 3: Extracting delta sequences..."
python src/extract_deltas.py \
    --collected "$DATA_DIR/collected/collected_examples.json" \
    --output "$DATA_DIR/collected/delta_sequences.json" \
    --model "$MODEL" \
    --device "$DEVICE"

# Step 4: Check data stats
echo ""
echo "Step 4: Data statistics..."
python -c "
import json
with open('$DATA_DIR/collected/delta_sequences.json') as f:
    data = json.load(f)

total = len(data)
followed = sum(1 for d in data if d['followed_nudge'])
by_type = {}
for d in data:
    t = d['nudge_type']
    if t not in by_type:
        by_type[t] = {'total': 0, 'followed': 0}
    by_type[t]['total'] += 1
    by_type[t]['followed'] += int(d['followed_nudge'])

print(f'Total examples: {total}')
print(f'Followed nudge: {followed} ({100*followed/total:.1f}%)')
print()
print('By type:')
for t, s in by_type.items():
    print(f'  {t}: {s[\"followed\"]}/{s[\"total\"]} followed ({100*s[\"followed\"]/max(1,s[\"total\"]):.1f}%)')
"

# Step 5: Train probes (optional validation)
echo ""
echo "Step 5: Training probes (validation)..."
python src/train_probe.py \
    --data_dir "$DATA_DIR/collected" \
    --device "$DEVICE"

# Step 6: Train oracle
echo ""
echo "Step 6: Training trajectory oracle..."
python src/train_oracle.py \
    --data "$DATA_DIR/collected/delta_sequences.json" \
    --model "$MODEL" \
    --output_dir "checkpoints/oracle" \
    --epochs 5 \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "Experiment complete!"
echo "========================================"
