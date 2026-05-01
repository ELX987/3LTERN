pretrain_ternary_llm.py

Pretraining scaffold for dense or MoE W1.58A8 ternary LLMs using EL_ternCUDA_kernel.cu.

It provides:
  - JSON / JSONL, Markdown, Parquet / .pq ingestion.
  - Hugging Face dataset fetching/streaming, including chat/conversation schema auto-detection.
  - Optional optimized JSON/Parquet/HF column with pre-tokenized token IDs.
  - Fast batched tokenization into a contiguous uint32 memmap cache.
  - Dense or MoE decoder-only Transformer with BitLinear layers.
  - Custom CUDA BitLinear calls through a small PyTorch C++ extension wrapper.
  - FP32 trainable/shadow weights, STE backward, W1.58 ternary packed safetensors export.
  - PyTorch SDPA/FlashAttention-compatible attention, optional torch.compile, DDP/FSDP.
  - Two-stage ternary QAT schedule: high LR/weight decay then cooldown with zero weight decay.

Important: tokenization produces integer token IDs. The 1.58-bit ternary representation is for
BitLinear weights; activations are quantized to INT8 by the custom CUDA kernel.

Dense local-data example:
  python pretrain_ternary_llm.py \
    --data 'data/*.jsonl' 'data/**/*.md' 'data/*.parquet' \
    --tokenizer ./tokenizer --output-dir runs/ternary_dense \
    --architecture dense --target-params 350M --layers 24 --seq-len 2048 \
    --batch-size 2 --grad-accum-steps 16 --max-steps 200000 \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h \
    --use-custom-kernel --compile


HF streaming example for the GLM-5.1 reasoning dataset:
  python pretrain_ternary_llm.py \
    --hf-dataset Jackrong/GLM-5.1-Reasoning-1M-Cleaned --hf-split train \
    --hf-streaming --stream-train --tokenizer auto \
    --output-dir runs/ternary_hf_stream --architecture dense --target-params 350M \
    --layers 24 --seq-len 2048 --batch-size 2 --grad-accum-steps 16 --max-tokens 10B \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h

MoE example:
  python pretrain_ternary_llm.py \
    --data 'data/train.parquet' --text-column text \
    --tokenizer ./tokenizer --output-dir runs/ternary_moe \
    --architecture moe --layers 24 --hidden-size 2048 --heads 16 \
    --num-experts 16 --top-k 2 --moe-num-layers 12 --moe-layer-stride 2 \
    --batch-size 1 --grad-accum-steps 32 --max-steps 200000 \
    --kernel-cu ./EL_ternCUDA_kernel.cu --kernel-header ./EL_ternCUDA_kernel.h \
    --use-custom-kernel
"""

* EL_ternCUDA_kernel.cu
 *
 * Native W1.58A8 BitLinear CUDA training primitive for PyTorch extension use.
 * Build compatibility patch: no the cuBLAS development header and no libcublas link dependency.
 *
 * Public layout expected by EL_ternCUDA_kernel.h:
 *   X          : __half  row-major [M, N]
 *   W_shadow   : float   row-major [K, N]     // K = output features
 *   W_packed   : uint32  row-major [K, ceil(N / 16)]
 *   W_scale    : float   [K]                  // absmean scale per output row
 *   Y          : __half  row-major [M, K]
 *   dY         : __half  row-major [M, K]
 *   dX         : __half  row-major [M, N]
 *   dW         : float   row-major [K, N]
 *
 * Ternary encoding, 16 weights per uint32:
 *   00 ->  0
 *   01 -> +1
 *   10 -> -1
 *   11 ->  0 / reserved
 * This is branchless sign/mask decoding: q = bit0 - bit1.
 *
 * Notes:
 *   - Forward path: activation FP16 -> per-row INT8, ternary W packed -> int8 lanes,
 *     __dp4a -> INT32 accumulation, dequant to FP16.
 *   - dX is a correctness-oriented low-bit kernel.
 *   - dW uses a self-contained CUDA fallback kernel; the training script defaults to
 *     PyTorch/cuBLAS for dW outside this extension so minimal CUDA images do not need
 *     cuBLAS development headers during extension compilation.
