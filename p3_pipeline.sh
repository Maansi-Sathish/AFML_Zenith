#!/usr/bin/env bash
# Person 3 pipeline for Kaggle. Run one stage per notebook cell:
#   !bash p3_pipeline.sh setup | data | time | lr | final | bench
# Env overrides: TRIALS (default 5), MINDUR seconds per trial (default 20).
set -e
DATA=/kaggle/working/data
RUNS=/kaggle/working/runs
RES=/kaggle/working/results
TRIALS=${TRIALS:-5}
MINDUR=${MINDUR:-20}
mkdir -p "$RUNS" "$RES"

best_lr () {  # $1 = arch -> prints lr with the lowest best_val_ppl among $RUNS/$1_lr*
python - "$1" <<'EOF'
import glob, json, sys
a = sys.argv[1]
best = None
for p in glob.glob(f"/kaggle/working/runs/{a}_lr*/result.json"):
    r = json.load(open(p)); lr = p.split(f"{a}_lr")[1].split("/")[0]
    print(f"  {a} lr={lr}: best_val_ppl={r['best_val_ppl']:.2f}", file=sys.stderr)
    if best is None or r["best_val_ppl"] < best[0]:
        best = (r["best_val_ppl"], lr)
print(best[1])
EOF
}

case "$1" in
setup)
  pip install -q tokenizers datasets ncps codecarbon nvidia-ml-py
  ;;
data)  # needs HF_TOKEN in the environment (Kaggle secret) and the dataset licence accepted
  python prep_data.py --out_dir "$DATA"
  ;;
time)  # 0.5M-token timing run -> estimate for 60M tokens (x120)
  for a in lstm cfc; do
    python train_text.py --arch $a --data_dir "$DATA" --out_dir "$RUNS/${a}_time" \
      --max_tokens 500000 --eval_every_tokens 250000 --lr 1e-3
    python - "$a" <<'EOF'
import json, sys
a = sys.argv[1]
r = json.load(open(f"/kaggle/working/runs/{a}_time/result.json"))
s = r["train_time_s"]
print(f"== {a}: {s:.0f}s for 0.5M tokens -> ~{s*120/3600:.1f} h for 60M tokens (train time only)")
EOF
  done
  ;;
lr)    # 3 learning rates x 2 models, 2M tokens each; picks the best on validation perplexity
  for a in lstm cfc; do
    for lr in 3e-4 1e-3 3e-3; do
      python train_text.py --arch $a --data_dir "$DATA" --out_dir "$RUNS/${a}_lr${lr}" \
        --max_tokens 2000000 --eval_every_tokens 500000 --lr $lr
    done
    best_lr $a > "$RUNS/best_lr_$a.txt"
    echo "== best lr for $a: $(cat $RUNS/best_lr_$a.txt)"
  done
  ;;
final) # 60M-token runs. Usage: bash p3_pipeline.sh final lstm   (or cfc)
  a=$2; lr=$(cat "$RUNS/best_lr_$a.txt")
  python train_text.py --arch $a --data_dir "$DATA" --out_dir "$RUNS/${a}_final" \
    --max_tokens 60000000 --seq_len 256 --batch 32 --lr $lr --resume
  # if the notebook crashes, rerun the same command: --resume continues from ckpt_full.pt
  mkdir -p results/text/runs/$a && cp "$RUNS/${a}_final/log.csv" "$RUNS/${a}_final/result.json" results/text/runs/$a/
  echo "== copied logs to results/text/runs/$a/ (git add + push them)"
  ;;
bench) # energy per step: train + decode. Run with nothing else on the GPU.
  B="python bench_lm.py --trials $TRIALS --min_duration $MINDUR"
  $B --arch lstm --impl optimized --phases train --seq_lens 256 --batch_sizes 32 --out_csv $RES/bench_train.csv
  $B --arch lstm --impl naive     --phases train --seq_lens 256 --batch_sizes 32 --out_csv $RES/bench_train.csv
  $B --arch transformer --impl optimized --phases train --seq_lens 256 --batch_sizes 32 --out_csv $RES/bench_train.csv
  $B --arch transformer --impl naive     --phases train --seq_lens 256 --batch_sizes 32 --out_csv $RES/bench_train.csv
  $B --arch cfc --phases train --seq_lens 256 --batch_sizes 32 --out_csv $RES/bench_train.csv
  for a in lstm cfc transformer; do
    $B --arch $a --phases decode --seq_lens 128,512,2048 --batch_sizes 1,4 --out_csv $RES/bench_decode.csv
  done
  ;;
*)
  echo "usage: bash p3_pipeline.sh setup|data|time|lr|final <arch>|bench"; exit 1
  ;;
esac