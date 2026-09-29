#!/usr/bin/env bash
# Revision R1 — after all shards finished: analyses, figures, side checks, bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
export TCM_ROOT="$HOME/tcm_3class"
export TCM_DATA="$HOME/tcm_3class/outputs"
export RES_DIR="$PWD/outputs"
export PYTHONIOENCODING=utf-8
PY="$TCM_ROOT/.venv/bin/python"

n_done=$(wc -l < "$RES_DIR/shards_done.txt" 2>/dev/null || echo 0)
[ "$n_done" -ge 4 ] || { echo "only $n_done/4 shards done — refusing to analyse partial results"; exit 1; }

echo "== SMOTE family on mu-sorted features (CPU, ~10 min) =="; "$PY" rerun_smote_musorted.py > "$RES_DIR/log_smote_musorted.txt" 2>&1 || echo "smote rerun failed (see log)"
echo "== master statistics ==";        "$PY" analyze_master.py            > "$RES_DIR/log_analyze_master.txt" 2>&1
echo "== subject-level (leakfree) =="; "$PY" analyze_subject_level.py --protocol leakfree > "$RES_DIR/log_subject_leakfree.txt" 2>&1
echo "== grad-cam resnet_gc ==";       "$PY" analyze_gradcam_all_heldout.py --model resnet_gc > "$RES_DIR/log_gradcam_gc.txt" 2>&1
echo "== grad-cam resnet ==";          "$PY" analyze_gradcam_all_heldout.py --model resnet    > "$RES_DIR/log_gradcam_resnet.txt" 2>&1
echo "== fig 7 ==";                    "$PY" plot_model_comparison.py      > "$RES_DIR/log_plot.txt" 2>&1
echo "== discard counts (read-only pass over the PNGs) =="; "$PY" count_discards.py > "$RES_DIR/log_discards.txt" 2>&1 || echo "count_discards failed (see log)"
echo "== synthetic round-trip in the server environment (isolated output dir) =="
mkdir -p "$RES_DIR/roundtrip_server"
sed -e "s#^OUTPUT_DIR = ROOT / \"outputs\"#OUTPUT_DIR = Path('$RES_DIR/roundtrip_server')#" "$TCM_ROOT/src/validate_roundtrip.py" > "$RES_DIR/roundtrip_server/validate_roundtrip_copy.py"
( cd "$TCM_ROOT" && PYTHONPATH="$TCM_ROOT" "$PY" "$RES_DIR/roundtrip_server/validate_roundtrip_copy.py" > "$RES_DIR/roundtrip_server/log.txt" 2>&1 ) || echo "round-trip rerun failed (see log)"

echo "== bundle (weights excluded; they stay on the server) =="
tar -C "$PWD" --exclude='outputs/weights' -czf "$HOME/revision_r1_results_$(date +%Y%m%d_%H%M).tgz" outputs
ls -la "$HOME"/revision_r1_results_*.tgz | tail -1
