# Memory-Traffic-First BitNet Plan

## 목적

흙수저용 LLM의 핵심 병목은 "GPU가 충분한가"보다 "토큰 하나를 만들 때 메모리를 얼마나 읽고 쓰는가"다. 작은 장비에서는 연산량보다 weight, KV cache, activation scratch를 DRAM/VRAM/CPU cache 사이에서 옮기는 시간이 더 쉽게 병목이 된다.

따라서 이 프로젝트의 우선 목표를 다음처럼 정의한다.

```text
목표: parameters를 줄인다
보다
목표: bytes moved per generated token을 줄인다
```

BitNet b1.58은 이 목표에 맞다. 단, 조건이 있다. PyTorch에서 ternary weight를 매번 float tensor로 만들고 `linear()`에 넣는 방식은 packed ternary kernel이 아니므로 메모리 이동을 줄이지 못한다. 오히려 원본 weight read, ternary temp write, ternary temp read가 추가되어 더 느려질 수 있다.

## 핵심 지표

1차 지표:

```text
bytes_per_token = weight_read_bytes
                + kv_cache_read_bytes
                + kv_cache_write_bytes
                + activation_scratch_bytes
                + dequant_or_repack_bytes
```

2차 지표:

```text
tokens_per_second ~= memory_bandwidth / bytes_per_token
```

정확한 성능은 kernel, cache hit, batch, prefill/decode 비율에 따라 달라지지만, bytes/token이 큰 방향은 흙수저용 추론에서 오래 버티기 어렵다.

## Decode 단계의 주요 traffic

batch 1, autoregressive decode 기준으로 layer 하나에서 대략 다음 traffic이 발생한다.

### 1. Weight streaming

일반 LLaMA linear projection:

```text
weight_elements_layer =
    q_proj:    H * H
  + k_proj:    H * KV
  + v_proj:    H * KV
  + o_proj:    H * H
  + gate_proj: H * I
  + up_proj:   H * I
  + down_proj: I * H

KV = num_key_value_heads * head_dim
```

fp16/bf16 baseline:

```text
weight_read_bytes = weight_elements_layer * 2
```

native packed b1.58 target:

```text
weight_read_bytes ~= weight_elements_layer * 2 / 8
```

실제 b1.58은 log2(3) ~= 1.58 bits지만, 구현에서는 2-bit packing이 1차 목표다. scale/metadata는 별도지만 weight matrix 본체보다 작아야 한다.

### 2. KV cache streaming

decode token마다 과거 context의 K/V를 읽는다.

```text
kv_read_bytes_layer =
  batch * seq_len * KV * 2 * kv_dtype_bytes

kv_write_bytes_layer =
  batch * KV * 2 * kv_dtype_bytes
```

context가 길어지면 KV cache read가 weight read보다 커질 수 있다. 짧은 context에서는 weight streaming이 더 중요하고, 긴 context에서는 KV cache 압축이나 attention kernel이 중요해진다.

### 3. Activation and scratch

hidden state, MLP intermediate, attention scores/softmax scratch도 traffic을 만든다. 하지만 batch 1 decode에서는 보통 weight와 KV가 먼저 큰 항이다. MLP intermediate를 materialize하지 않고 fused SwiGLU/down projection으로 줄이면 도움이 된다.

## 현재 코드의 문제

현재 `BitLinear` 계열은 연구용 reference에 가깝다.

문제:

- packed ternary weight를 직접 matmul하지 않는다.
- forward마다 ternary weight tensor를 만든다.
- activation quantization도 별도 tensor를 만들어 traffic을 추가한다.
- `BitLinearOptimized`의 int8 snapshot은 실제 matmul path에 쓰이지 않는다.

따라서 현재 구조로는 "메모리 이동을 줄이는 BitNet"이 아니다. 올바른 목표는 다음이다.

```text
나쁜 path:
fp16 weight -> ternary temp tensor -> fp16/bf16 matmul

좋은 path:
packed ternary weight -> fused ternary matmul/add/sub kernel -> output
```

## 우선순위

### P0: bytes/token estimator

먼저 모델 크기, context 길이, dtype, packing policy에 따른 traffic을 숫자로 본다.

산출물:

- `scripts/estimate_memory_traffic.py`
- `reports/memory_traffic_*.json`

판정:

- 현재 PyTorch BitLinear가 baseline보다 traffic을 줄이는지 확인한다.
- packed b1.58 kernel이 어느 정도 이론적 이득을 주는지 확인한다.
- context 길이에 따라 weight bottleneck과 KV bottleneck이 바뀌는 지점을 찾는다.

### P1: inference-only packed weight format

학습용 `weight`와 추론용 packed ternary weight를 분리한다.

산출물:

- `pack_ternary_weight(weight) -> codes, scales`
- `unpack_ternary_weight(codes, scales)` for correctness only
- storage estimate and round-trip tests

주의:

- unpack은 test/debug용이다.
- 실제 inference hot path에서 full float weight를 materialize하면 실패다.

### P2: tiny packed linear prototype

처음부터 CUDA/Metal kernel로 가지 않는다. 먼저 CPU numpy/torch reference로 correctness를 확인한다.

목표:

- packed code를 읽어서 add/sub accumulation으로 output을 만든다.
- matmul 결과가 `alpha*T` reference와 가까운지 확인한다.
- traffic model과 실제 runtime trend가 같은지 본다.

### P3: fused hot path

진짜 절감은 여기서 시작된다.

후보 fusion:

- RMSNorm + ternary linear
- gate/up projection + SwiGLU
- down projection
- attention q/k/v projection
- KV write

원칙:

- full dequantized weight tensor를 만들지 않는다.
- MLP intermediate를 되도록 오래 저장하지 않는다.
- 반복적으로 같은 tensor를 DRAM에 쓰고 다시 읽지 않는다.

### P4: KV cache compression

context가 길어져 KV가 bottleneck이 되면 그때 KV cache를 본다.

후보:

- int8 KV cache
- 4-bit KV cache
- TurboQuant/RateQuant/HyperQuant 계열

우선순위:

- 짧은 context: packed weight kernel
- 긴 context: KV cache compression

## 실험 판정 기준

어떤 구현도 다음을 만족하지 못하면 흙수저용 목표와 거리가 멀다.

```text
1. bytes/token이 fp16 baseline보다 줄어든다.
2. full dequantized weight materialization이 hot path에 없다.
3. context 길이별 bottleneck이 수치로 설명된다.
4. speedup 주장이 bandwidth model과 크게 어긋나지 않는다.
5. 품질 측정 전에도 traffic 상으로 의미가 있어야 한다.
```

## 다음 행동

1. `estimate_memory_traffic.py`로 현재 config의 weight/KV traffic을 계산한다.
2. "current PyTorch BitLinear", "fp16 baseline", "packed b1.58 target"을 비교한다.
3. 결과를 보고 packed weight path가 먼저인지 KV compression이 먼저인지 정한다.

## Current Estimates

The estimator is implemented here:

```bash
.venv/bin/python scripts/estimate_memory_traffic.py --json-out reports/memory_traffic_bitllama_512x4.json
```

For the current `bitllama-110M-config`-style shape (`hidden=512`, `layers=4`), context length 2048:

```text
fp16 baseline                  ~= 49 MB/token
current PyTorch BitLinear      ~= 113 MB/token
packed b1.58 + fp16 KV         ~= 21 MB/token
packed b1.58 + int8 KV         ~= 13 MB/token
packed b1.58 + int4 KV         ~= 9 MB/token
```

Interpretation:

```text
The current PyTorch BitLinear path is a correctness/reference path, not a
low-resource inference path. It materializes extra tensors and moves more data
than fp16. The useful path starts at packed ternary weights read directly by a
hot inference kernel.
```

For longer contexts, KV read becomes the dominant term after weight packing, so
KV int8/int4 and later TurboQuant-style cache compression should be evaluated
after the packed-weight path exists.
