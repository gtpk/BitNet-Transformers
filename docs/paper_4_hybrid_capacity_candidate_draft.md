# Paper 4 Draft: Mostly-I2_S Models With Auxiliary Capacity

Status: position/candidate draft. Not a result paper yet.

Related skeleton: [Paper 4 Skeleton](./paper_4_hybrid_capacity_candidate.md)  
Central evidence: [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-4-hybrid-capacity-candidate-evidence)

## Abstract

Pure all-I2_S conversion is attractive because it preserves the storage and
memory-traffic benefits of bitnet.cpp I2_S. However, if factual quality plateaus,
the product path may require a small amount of auxiliary capacity. We frame this not
as abandoning I2_S, but as a budgeted topology decision: keep I2_S as the trunk and
add the smallest helper that buys real behavior. Early cheap screens are mostly
negative. Post-hoc F16 restore breaks co-adaptation, whole/targeted LoRA sidecars do
not yet produce actionable gains, and EGROW identifies stable sensitivity hotspots
but not a working growth action. The paper becomes valid only if a train-from-start
hybrid or capacity pocket improves factual quality enough to remain on the final
Pareto frontier.

## 1. Motivation

Native b1.58 success does not imply that every FP checkpoint has enough capacity in
the same topology after conversion. One possible response is to add capacity:

```text
Wq = gamma*T + residual
```

or to use selected Q2/Q3/F16 pockets. But our experiments show that capacity cannot
be added blindly. The adapted all-ternary model becomes a co-adapted system; changing
late layers after training can make it worse.

## 2. Negative Evidence So Far

| branch | result | implication |
| --- | --- | --- |
| HYBRID-001A post-hoc F16 restore | all restores worse than all-I2_S | co-adaptation mismatch |
| SIDE-001 LoRA sidecar | no clear behavior gain | global low-rank residual not obvious |
| EGROW-001 | sensitivity locator stable | useful diagnostic |
| EGROW-002 | top-k sidecar <= random/none | locator not yet actionable |
| WSYNC/H-I2S/SIGMA/RHT | behavior remains collapsed | data-free geometry not enough |

These results do not prove hybrid is dead. They rule out the cheap, post-hoc
versions.

## 3. Candidate Method

A valid hybrid method must be trained from the start:

```text
choose auxiliary regions
train with I2_S trunk + helper active
measure size/speed/quality against Q2_K and PT2
```

Possible helpers:

```text
selected Q2/Q3 regions
multi-strip ternary
small residual sidecars
train-from-start late-layer precision
```

## 4. Required Success Condition

Hybrid is worth a paper only if:

```text
quality improves over the best all-I2_S recipe
and
storage/speed remain meaningfully better than Q2_K or other baselines.
```

If it only improves quality by becoming a normal quantized model, it leaves the I2_S
research question.

## References

- BitNet b1.58: <https://arxiv.org/abs/2402.17764>
- bitnet.cpp runtime: <https://arxiv.org/abs/2502.11880>
- QuIP / incoherence processing as extreme low-bit context: <https://arxiv.org/abs/2307.13304>
- QuIP# / Hadamard and lattice codebooks: <https://arxiv.org/abs/2402.04396>
- Internal evidence: [PC Negative Branch Map](./pc_negative_branch_map.md), [Evidence Ledger capacity track](../reports/EVIDENCE_LEDGER.md#f-capacity--geometry-track----comprehensively-negative-at-160m-cost-ledger-rdt-001)
