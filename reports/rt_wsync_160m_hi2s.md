# RT-WSYNC H-I2S (rotation) -- 160M, data-free. Rotation FAILS too -> WSYNC track demoted.

model=Felladrin/Llama-160M-Chat-v1  eval=WikiText 60k tok  FACT panel 27 (rep-penalty 1.2)
H-I2S honest reference: y = Q_I2S(W H^T)(H x), block Hadamard tile 128 on the input activation
(NOT folded into a dense weight).

| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp | 3.0881 | 21.9 | 0.296 | 0 | 0 |
| pt (per-tensor b1.58) | 11.6401 | 113559 | 0.000 | 0.00070 | 0.668 |
| group (best scaling) | 9.6222 | 15096 | 0.000 | 0.00065 | 0.664 |
| **h_i2s (Hadamard rotation)** | **12.4852** | **264391** | **0.000** | **0.00069** | 0.670 |

## Verdict: rotation FAILS -> close the data-free WSYNC track

H-I2S (block-Hadamard tile 128) is WORSE than plain per-tensor (CE 12.49 vs 11.64, ppl 264k vs
113k) and does not even reduce weight MSE (0.00069 ~= pt 0.00070). The rotation neither improves
reconstruction nor lifts FACT (0.0). So on a non-STE 160M model, ROTATING the weights before
data-free ternary quant does not help; the activation rotation just adds error on top of an already
collapsed ternary weight. (The auto-verdict reads "PARTIAL" only because group-SCALING lowers CE
between two collapsed states; that is not behavioural recovery -- FACT is 0.0 for every transform,
and the rotation arm we actually tested is the worst.)

Clean negative, exactly as hoped:
```
scaling failed  (row/group/row_norm: CE moves between collapsed states, FACT 0.0)
rotation failed (h_i2s: worse than per-tensor, no MSE gain, FACT 0.0)
=> data-free weight-only sync does NOT rescue b1.58 collapse at 160M
=> the lever is representative data (PopQA blend) / adaptation / capacity, not weight geometry
```

Plan decision S4: do not spend more time on data-free weight-only sync. (Caveat: this tests the
DATA-FREE premise on a NON-STE model; rotation could still matter combined with STE/activation-aware
training -- but that is no longer "weight-only sync", it is the adaptation/objective lever we are
already pursuing with FACT-003H.) Claim discipline held: no speed claim, quality/reference only.
