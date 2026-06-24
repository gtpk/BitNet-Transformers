# GGUF / bitnet.cpp Export Scoping Plan

Document position: [Index](./index.md) -> next track after the packed reference ladder completed.

Related docs:

- [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)
- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Research Signal Note](./research_signal_note.md)

## Purpose

Phase 1-4 proved the Python reference ladder:

```text
packed storage -> model export/import -> packed runtime module -> blocked dequant matmul
```

The remaining unsolved part is latency. The Python reference path proves that
dense weight materialization can be avoided, but it is slower than dense matmul.
The next high-leverage question is:

```text
Can this project's groupwise alpha*T ternary format be exported into an
existing optimized ternary runtime before writing a custom kernel?
```

## Why This Before A Custom Kernel

Direct CPU/Metal/CUDA kernels are possible, but they are a different engineering
track. Export scoping can answer several questions first:

- whether an existing runtime format can represent this project's ternary
  `T in {-1,0,+1}` plus groupwise `alpha`
- whether logit equality survives outside the Python module path
- whether real latency and memory improve in an existing inference stack
- whether a custom kernel is still necessary after export experiments

## Non-Goals

- Do not assume the current bitnet.cpp or GGUF format without checking upstream
  source/docs first.
- Do not claim latency improvement until an exported artifact runs in an
  optimized runtime.
- Do not change the conversion algorithm to fit an export format before
  measuring quality impact.

## Scoping Steps

### Step 0: Inspect Current Runtime Format

Verify the current bitnet.cpp/GGUF expectations:

- tensor names and layer mapping
- ternary/b1.58 encoding layout
- scale granularity and dtype
- metadata requirements
- supported model architectures
- available logit or generation test path

Output: a compatibility table against `PackedTernaryWeight`.

### Step 1: Mapping Decision

Compare this project format:

```text
T packed as trit bytes
alpha shape = [out_features, n_groups]
grouping along input dimension
reconstruction = alpha[:, group] * T[:, input]
```

against the target runtime format.

Decision:

- direct mapping
- transform with lossless re-layout
- transform with lossy re-quantization
- blocked: custom kernel/export format needed

### Step 2: Minimal Export Artifact

Start with a tiny Llama-shaped fixture model before pretrained models.

Pass criteria:

- exported artifact loads
- target linears are represented by the export path
- non-target tensors remain fp16 or expected runtime dtype
- metadata round-trip is inspectable

### Step 3: Correctness Gate

Compare:

```text
Python S1-converted model logits
Python PackedTernaryLinear logits
exported-runtime logits
```

Pass criteria:

- exact or numerically tiny logit delta on the tiny fixture
- no layer-name or transpose mismatch
- generation smoke finite and non-degenerate

### Step 4: Runtime Gate

Only after correctness:

- storage size
- load memory
- per-token latency
- long-context memory interaction with KV cache

## TC Draft

| ID | Area | Check | Pass criterion |
| --- | --- | --- | --- |
| EXPORT-001 | Format | current bitnet.cpp/GGUF format inspected | compatibility table exists |
| EXPORT-002 | Mapping | `PackedTernaryWeight` -> target layout | direct/lossless/lossy/blocked decision |
| EXPORT-003 | Tiny artifact | fixture model exports and loads | no loader error |
| EXPORT-004 | Logits | exported runtime logits vs Python reference | max error threshold recorded |
| EXPORT-005 | Storage | artifact size vs fp16 | report exact ratio |
| EXPORT-006 | Latency | runtime latency vs Python reference/dense | report, no overclaim |

## Decision Rule

Proceed to export implementation if format mapping is direct or lossless.

If mapping requires lossy re-quantization, pause and run a quality gate before
claiming compatibility.

If mapping is blocked, split into:

1. custom GGUF extension/export path
2. custom CPU/Metal/CUDA fused kernel track
3. bitnet.cpp contribution or adapter track
