#!/usr/bin/env bash
# Revision R1 — launch the full repeated-CV rerun on ailab_server (~/tcm_3class).
#   * both protocols (leak-free + optimistic) so Table 5 and Table 6 share one environment
#   * per-beat probabilities + ResNet1D / ResNet1D+GC fold weights saved (subject-level
#     metrics, aggregated confusion matrices, Grad-CAM on all held-out beats)
#   * classical route on mu-sorted Gaussian features (Reviewer 1, Major 4)
# Data (~/tcm_3class/outputs/*.npy|csv) is READ ONLY; everything is written to revision_r1/outputs.
set -euo pipefail
cd "$(dirname "$0")/.."                      # ~/tcm_3class/revision_r1
export TCM_ROOT="$HOME/tcm_3class"
export TCM_DATA="$HOME/tcm_3class/outputs"
export RES_DIR="$PWD/outputs"
export TORCH_THREADS="${TORCH_THREADS:-3}"
PY="$TCM_ROOT/.venv/bin/python"
mkdir -p "$RES_DIR"

echo "== environment ==" | tee "$RES_DIR/env_info.txt"
{ hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc;
  "$PY" -c "import sys,torch,numpy,scipy,sklearn,xgboost,pandas,matplotlib;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,'pandas',pandas.__version__,'matplotlib',matplotlib.__version__)";
} | tee -a "$RES_DIR/env_info.txt"
"$PY" -m pip freeze > "$RES_DIR/pip_freeze.txt"
md5sum "$TCM_DATA"/beats_matrix.npy "$TCM_DATA"/dataset.csv "$TCM_DATA"/tight_augmented_beats.npy "$TCM_DATA"/tight_augmented_meta.csv > "$RES_DIR/data_md5.txt"

# 4 shards over the 10 partition seeds; each shard runs its repeats sequentially.
launch() {  # $1 = shard name, rest = repeats
  local name=$1; shift
  nohup bash -c "for r in $*; do $PY rerun_deep_with_preds.py --only-repeat \$r --protocols leakfree,optimistic --save-weights resnet,resnet_gc > $RES_DIR/log_deep_r\$r.txt 2>&1; done; echo SHARD_${name}_DONE >> $RES_DIR/shards_done.txt" \
       > "$RES_DIR/nohup_$name.txt" 2>&1 &
  echo "shard $name (repeats $*) pid $!"
}
rm -f "$RES_DIR/shards_done.txt"
launch A 0 4 8
launch B 1 5 9
launch C 2 6
launch D 3 7
echo "started $(date). Monitor: tail -n1 $RES_DIR/log_deep_r*.txt ; done when shards_done.txt has 4 lines."
