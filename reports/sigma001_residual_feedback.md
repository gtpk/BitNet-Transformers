# SIGMA-001 residual-feedback ternary projection

model=Felladrin/Llama-160M-Chat-v1  eval=WikiText 60000 tok  FACT panel=27

| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp | 3.0881 | 21.93 | 0.296 | 0.0 | 0.0 | {'ok': 20, 'salad': 1, 'empty': 6} |
| pt | 11.6401 | 113559.27 | 0.0 | 0.000695 | 0.6682 | {'ok': 26, 'salad': 1} |
| sigma_row_a0.5 | 13.3141 | 605695.52 | 0.037 | 0.000794 | 0.6734 | {'ok': 25, 'empty': 1, 'loop': 1} |
| sigma_g128_a0.5 | 12.3986 | 242466.41 | 0.0 | 0.000777 | 0.6724 | {'ok': 27} |
| sigma_g128_a1.0 | 13.2846 | 588083.85 | 0.0 | 0.001036 | 0.6849 | {'ok': 26, 'loop': 1} |

best_fact=sigma_row_a0.5  best_ce=pt

VERDICT: FAIL

PASS requires FACT movement, not only CE movement inside a collapsed regime.
