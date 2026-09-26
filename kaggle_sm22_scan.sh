#!/bin/bash
# SM22 2.5D seed scan on Kaggle 2xT4, fp32, batched: each GPU advances half the seeds together
# in one vmapped step (sm22_batch.py). fp32 parity vs fp64 verified at N=64 (seeds 1-2):
# maxgrad rel diff <=5e-4 for A<=0.9, Lambda99 peak identical to 3 s.f.
# Kaggle notebook cells (setup as in kaggle_grow.ipynb):
#   !pip install -q -U "jax[cuda12]"
#   !git clone https://github.com/alfredmallet/jax_constantB.git /kaggle/working/jax_constantB
#   %cd /kaggle/working/jax_constantB
#   !bash kaggle_sm22_scan.sh 8-107 128 1.2
SEEDS=${1:-8-57}; N=${2:-128}; AMAX=${3:-1.2}
lo=${SEEDS%-*}; hi=${SEEDS#*-}; mid=$(( (lo + hi) / 2 ))
OUT=/kaggle/working/sm22_scan_N$N; mkdir -p $OUT
export SM22_FP32=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=0 python3 -u sm22_batch.py --N $N --seeds $lo-$mid       --Amax $AMAX --out $OUT > $OUT/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u sm22_batch.py --N $N --seeds $((mid+1))-$hi --Amax $AMAX --out $OUT > $OUT/gpu1.log 2>&1 &
wait
cd /kaggle/working && tar czf sm22_scan_N$N.tgz sm22_scan_N$N
