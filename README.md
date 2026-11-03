# LM_Dec_Long_Memorizing

Memorizing - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2022_Memorizing-transformers-pytorch
- Upstream / license: lucidrains/memorizing-transformers-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- segments of 128 inside each 512 window (upstream: 5 segments of 512); kNN layer 4, XL layers 5-6 with 128 memories; faiss-cpu instead of faiss-gpu

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 16.42 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.777 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.770 / test 1.740 bits per byte (time-capped)
- Peak GPU memory at length 2048 (batch 1): train 499 MB, inference 57 MB
- Peak GPU memory at length 8192 (batch 1): train 1876 MB, inference 60 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.02 (keys removed 0.02); 4 pairs @ 2048: 0.01 (keys removed 0.02); 16 pairs @ 512: 0.02 (keys removed 0.01); 16 pairs @ 2048: 0.01 (keys removed 0.01)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check memorizing
python run.py lm02 memorizing      # enwik8 is downloaded to data/ on first use
```
