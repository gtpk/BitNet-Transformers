# RHT-002 dithered/randomized Hadamard reference

model=Felladrin/Llama-160M-Chat-v1  tile=128 seed=41 dither=0.25

| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp | 3.0881 | 21.93 | 0.296 | 0.0 | 0.0 | {'ok': 20, 'salad': 1, 'empty': 6} |
| pt | 11.6401 | 113559.27 | 0.0 | 0.000695 | 0.6682 | {'ok': 26, 'salad': 1} |
| h1 | 12.4852 | 264391.53 | 0.0 | 0.00069 | 0.6699 | {'ok': 24, 'loop': 3} |
| rht1 | 12.4852 | 264391.53 | 0.0 | 0.00069 | 0.6699 | {'ok': 24, 'loop': 3} |
| rht2 | 11.4217 | 91283.73 | 0.0 | 0.00069 | 0.6698 | {'loop': 16, 'ok': 11} |
| rht1_dither | 12.6866 | 323370.1 | 0.0 | 0.000694 | 0.6711 | {'ok': 27} |

best_fact=rht2  best_ce=rht2

VERDICT: FAIL

PASS requires FACT movement. CE-only movement inside a collapsed regime is partial at best.
