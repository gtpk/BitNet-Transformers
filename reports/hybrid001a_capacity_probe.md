# HYBRID-001A late-layer capacity probe (FACT-003C lambda=0.2 latents)

model=TinyLlama-1.1B-Chat  ckpt=fact003c_mixed_ckl0.2/ckpt.pt  panel=27 prompts
Restore = run those target linears in FP (STE latent) instead of ternary; rest stay b1.58.
PyTorch fact_rate (rep-penalty 1.2, greedy, contains-match); A0 reproduces the I2_S baseline.

| arm | restore | FP params | fact_rate | CE | ppl | tags |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A0 | none (all I2_S) | 0.0M | 0.148 | 3.840 | 46.5 | {'ok': 27} |
| A1 | last 1 block | 44.0M | 0.074 | 4.935 | 139.1 | {'ok': 26, 'loop': 1} |
| A2 | last 2 blocks | 88.0M | 0.037 | 4.731 | 113.5 | {'ok': 27} |
| A3 | last 4 blocks | 176.0M | 0.111 | 4.555 | 95.1 | {'ok': 26, 'salad': 1} |
| A4 | last 4 blocks attn only | 38.0M | 0.111 | 5.404 | 222.2 | {'ok': 17, 'repetitive': 6, 'loop': 4} |
| A5 | last 4 blocks MLP only | 138.0M | 0.037 | 4.971 | 144.2 | {'salad': 4, 'empty': 4, 'ok': 19} |

A0 baseline fact_rate = 0.148; best arm = A0 @ 0.148 (restoring late layers to FP HURTS).

VERDICT: facts barely move (every restore arm is <= A0, and CE gets WORSE) -> NOT mainly a
capacity bottleneck. The model was STE-trained with ALL target linears ternary, so early
layers co-adapted to feed ternary late layers; post-hoc un-quantizing late layers to their FP
latent breaks that coherence (distribution mismatch) -> fact & CE both degrade. This rules out
the cheap post-hoc capacity intervention; it does NOT rule out a train-from-start hybrid (where
early layers co-adapt to FP late layers). Next per the decision tree: objective/data --
lm_head unfreeze, protected factual replay, content-AKL.