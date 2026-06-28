# HOME-001 activation homeostasis smoke (160M)

model=Felladrin/Llama-160M-Chat-v1  recipe=content-KL 0.2 + PopQA blend 5%  steps=300
layers=last rho=1.0

| arm | eta | eval_panel | popqa_tight | popqa_train | CE | recovered | tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| home_eta0 | 0.0 | 0.111 | 0.04 | 0.087 | 4.093 | 0.8853925206696517 | {'ok': 27} |
| home_eta0p01 | 0.01 | 0.111 | 0.03 | 0.062 | 4.084 | 0.8862339008340582 | {'ok': 27} |
| home_eta0p05 | 0.05 | 0.111 | 0.025 | 0.037 | 4.092 | 0.8858757410303596 | {'ok': 27} |

best=home_eta0 delta_eval_vs_eta0=+0.000

VERDICT: NO CLEAR SIGNAL -- homeostasis does not buy >=0.05 eval over eta=0 under this smoke.
