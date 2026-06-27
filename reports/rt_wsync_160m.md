# RT-WSYNC-001 160M weight-only b1.58 table (data-free, no training)

model=Felladrin/Llama-160M-Chat-v1  eval=WikiText 60k tok  FACT panel 27 (rep-penalty 1.2)
Sigma_x~=I (no calibration data): weight-geometry only.

| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp | 3.0881 | 21.9 | 0.296 | 0 | 0 | ok20 salad1 empty6 |
| pt (per-tensor b1.58) | 11.6401 | 113559 | 0.000 | 0.00070 | 0.668 | ok26 salad1 |
| row | 9.9785 | 21557 | 0.000 | 0.00066 | 0.661 | ok22 loop5 |
| group | 9.6222 | 15096 | 0.000 | 0.00065 | 0.664 | ok24 loop3 |
| row_norm | 11.2187 | 74508 | 0.000 | 0.00058 | 1.000 | ok25 loop2 |

## Honest verdict: data-free SCALING does NOT rescue ternary conversion

The script's auto-VERDICT says PASS because group beats per-tensor by ~2 nats CE (>=0.5 bar), but
that is MISLEADING: it is movement between two collapsed states (ppl 15096 vs 113559, both vs fp
21.9). The behaviour metric is the meaningful one and **every ternary arm has fact_rate 0.000** --
no arm clears the >=0.05 FACT bar. row_norm corrects row norms exactly (rnr 1.000) and still gives
ppl 74508 / fact 0.0, so even norm correction does nothing for behaviour.

So: **simple data-free weight transforms (row / group / row-norm scaling) collapse this non-STE
model just like plain per-tensor.** This is the expected PTQ-collapse the whole project started
from -- without STE training or representative data, ternary conversion destroys the model, and
re-distributing the per-tensor/row/group SCALE does not change that. Plan decision **S2/S4**:
Sigma_x~=I weight scaling is too weak; the lever is representative data (PopQA blend) / capacity.

## What this motivates: the ROTATION arm (H-I2S / WSYNC-004)

Scaling failing is exactly why QuaRot/QuIP use ROTATION, not scaling: a Hadamard rotation spreads
weight energy across coordinates (incoherence) so ternary has fewer outliers to destroy. H-I2S
(`Wq = Q(W H^T)`, apply `H x` at runtime; diagnostic effective weight `W_hat = Q(W H^T) H`, block
Hadamard tile 128) is the one remaining data-free lever and the decisive next falsification: add a
`hadamard` arm to this same table. Pass = hadamard beats per-tensor by >=0.5 nats CE AND moves FACT
off 0.0 (not just CE between broken states). If rotation also leaves FACT at 0.0, data-free weight
sync is dead for this model and the lever is purely data/capacity (PopQA blend).
