# EGROW-001 per-layer I2_S instability/bottleneck (160M, 2 seeds)

model=Felladrin/Llama-160M-Chat-v1  recipe=content-KL 0.2 + PopQA blend 5%  steps=200  (logger only)
B_l = mean(norm(H_time, flip, grad-conflict, update-reversal)) * norm(output_residual) * norm(task_saliency)

## First success condition: MET

- top-8 bottleneck layers overlap **7/8** across seeds 41 & 42 (stable ranking).
- NOT "last layers only": top blocks span {0, 3, 9, 10, 11} (12-block model).

Shared top bottleneck layers (both seeds):
```
layers.0.mlp.down_proj   layers.3.mlp.down_proj   layers.9.mlp.down_proj
layers.10.mlp.down_proj  layers.11.mlp.down_proj  layers.9.self_attn.o_proj
layers.11.self_attn.o_proj
```
down_proj dominates (5 of top-8) -- consistent with down_proj being a known outlier/sensitivity
hotspot in quantization (SmoothQuant/AWQ).

## Honest finding: the discriminator is SENSITIVITY, not temporal instability

For every top layer: **flip_rate ~= 0.000, temporal_entropy ~= 0.002** -- the ternary codes SETTLE,
they do NOT keep flipping. B_l is driven entirely by **output_residual (0.5-0.66) x task_saliency**.
So the original "STE keeps flipping => capacity bottleneck" intuition does NOT hold at 160M; the real
signal is HAWQ-style sensitivity (activation-weighted residual x task saliency). The plan already
hedged that flip-rate alone is not proof; the data confirms the instability factor is near-uniform
and the multiplicative B_l's residual*saliency factor does all the discriminating. (Net for the
method: keep the multiplicative bottleneck score, but expect sensitivity -- not instability -- to
locate growth sites.)

## Next: EGROW-002 (targeted sidecar)

SIDE-001 all-layer sidecar showed no clear FACT lever. EGROW-002 tests whether a rank-2/4 sidecar on
the **top-k by B_l (down_proj-heavy: layers 0/3/9/10/11)** beats a **random-k** sidecar of equal
bytes. If top-k > random-k and FACT moves, the bottleneck localization is real -> 1.1B EGROW-004.
If top-k == random-k, the localization does not buy anything and the lever stays data/objective.
