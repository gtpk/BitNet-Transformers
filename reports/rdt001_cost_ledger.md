# RDT-001 Cost Ledger

This ledger is a branch-prioritization smoke, not a quality claim.

| branch | arm | eval | Δeval | CE | ΔCE | extra bytes | eval gain / MB | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WSYNC-scaling | fp | 0.296 | 0.296 | 3.0881 | -8.552 | 0 | None | data-free |
| WSYNC-scaling | pt (per-tensor b1.58) | 0.0 | 0.0 | 11.6401 | 0.0 | 0 | None | data-free |
| WSYNC-scaling | row | 0.0 | 0.0 | 9.9785 | -1.6616 | 0 | None | data-free |
| WSYNC-scaling | group | 0.0 | 0.0 | 9.6222 | -2.0179 | 0 | None | data-free |
| WSYNC-scaling | row_norm | 0.0 | 0.0 | 11.2187 | -0.4214 | 0 | None | data-free |
| WSYNC-HI2S | fp | 0.296 | 0.296 | 3.0881 | -8.552 | 0 | None | data-free |
| WSYNC-HI2S | pt (per-tensor b1.58) | 0.0 | 0.0 | 11.6401 | 0.0 | 0 | None | data-free |
| WSYNC-HI2S | group (best scaling) | 0.0 | 0.0 | 9.6222 | -2.0179 | 0 | None | data-free |
| WSYNC-HI2S | h_i2s (Hadamard rotation) | 0.0 | 0.0 | 12.4852 | 0.8451 | 0 | None | data-free |
| SIDE-001 | rank0 | 0.185 | 0.0 | 4.058 | 0.0 | 0 | None | ok 26, repetitive 1 |
| SIDE-001 | rank2 | 0.222 | 0.037 | 4.04 | -0.018 | 847872 | 0.043639 | ok 25, salad 1, rep 1 |
| SIDE-001 | rank4 | 0.185 | 0.0 | 4.024 | -0.034 | 1695744 | 0.0 | ok 27 |
| SIDE-001 | rank8 | 0.185 | 0.0 | 4.031 | -0.027 | 3391488 | 0.0 | loop 1, ok 24, salad 1, empty 1 |
| EGROW-002 | none | 0.222 | 0.0 | 4.061 | 0.0 | 0 | None | ok25 loop1 rep1 |
| EGROW-002 | topk | 0.148 | -0.074 | 4.042 | -0.019 | 135168 | -0.547467 | ok25 rep2 |
| EGROW-002 | randk | 0.185 | -0.037 | 4.063 | 0.002 | 135168 | -0.273733 | ok26 rep1 |

## Verdict

NO LOW-COST POSITIVE: current PC branches do not buy >=0.05 eval; wait for FACT-003H or a new mechanism.
