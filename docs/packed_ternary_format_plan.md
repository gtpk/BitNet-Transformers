# Packed Ternary Weight Format Plan

Document position: [Index](./index.md) -> first artifact after real-text validation passed.

Related docs:

- [Real Tiny Text Validation Plan](./real_tiny_text_validation_plan.md)
- [Existing Model to BitNet Conversion Plan](./existing_model_to_bitnet_conversion_plan.md)
- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)

## Purpose

품질 검증은 끝났다. scaled-STE는 synthetic gate(seed/group-size/activation)와 real-text
(Wikitext, 3 seed)에서 projected-QAT를 이겼다. 다음 질문은 알고리즘이 아니라 저장이다:

```text
이론적 b1.58(=log2(3)=1.585 bits/elem)이 실제 저장/이동 byte 감소로 이어지는가?
```

이 문서는 packed ternary weight format을 정의하고, 실제 packing 구현보다 먼저
format/metadata/TC를 고정한다. kernel(matmul 최적화)은 이 format이 검증된 뒤에 온다.

## Non-goals

- 빠른 ternary matmul kernel. 여기서는 unpack -> dense reconstruction -> reference matmul만 한다.
- Metal/CUDA/bitnet.cpp 타겟 선택. CPU reference가 통과한 뒤에 판단한다.
- activation quantization 저장. 이 format은 weight 전용이다.

## Weight representation

`conversion.S1` / `ScaledBitLinear`와 동일한 groupwise-input 정책을 그대로 저장한다.

```text
W in R[out, in]
group_size G  (default 64, along input dim)
per group (out-row, input-block):
    threshold = lambda * mean(|W_block|)         # lambda default 0.7
    T = sign(W) where |W| > threshold else 0     # T in {-1,0,+1}
    alpha = mean(|W| over T != 0)                # >= 0, per (row, group)
reconstruction: W_approx[:, block] = alpha[:, g] * T[:, block]
```

저장 대상은 `T`(trit code)와 `alpha`(groupwise scale) 두 가지뿐이다.

## Packing schemes

| scheme | bits/elem | layout | 용도 |
| --- | --- | --- | --- |
| `two_bit` | 2.0 | trit 1개를 2bit, 1 byte당 4 trit | 단순/정렬, baseline |
| `trit` | 1.6 | base-3로 5 trit을 1 byte (3^5=243<=256) | b1.58 bound 근사, 기본 |

trit을 저장 전에 `{-1,0,+1} -> {0,1,2}`로 shift한다.

```text
two_bit byte = s0 | s1<<2 | s2<<4 | s3<<6                 # s_i in {0,1,2}, 값 3은 미사용
trit    byte = s0 + s1*3 + s2*9 + s3*27 + s4*81           # 0..242
```

마지막 블록은 0으로 padding 후 unpack 시 `numel`로 truncate한다. 측정된 trit 밀도는
**1.600 bits/elem**으로 이론 하한 1.585에 거의 도달한다(scale 제외).

## Storage layout / metadata

`PackedTernaryWeight` (per linear layer):

```text
scheme        : "two_bit" | "trit"
out_features  : int
in_features   : int
group_size    : int
n_groups      : ceil(in_features / group_size)
packed        : uint8[ceil(numel / k)]      # k = 4 (two_bit) or 5 (trit)
scales        : float[out_features, n_groups]   # alpha grid; fp16 for storage, fp32 in memory
```

- code는 row-major flatten 순서로 packing한다(`[out, in]` C-order).
- scale은 `[out_features, n_groups]` 2D layout. groupwise alpha를 그대로 들고 있어
  reconstruction이 row/group broadcast로 끝난다.
- 저장은 `torch.save(state_dict-like)` round-trip. 모델 단위는 layer key -> PackedTernaryWeight
  매핑으로 확장한다(다음 Phase).
- 현재 reference artifact는 안전한 round-trip을 위해 `torch.save` state를 사용한다. 저장량
  리포트는 추론 artifact 목표인 fp16 scale 기준으로 계산하며, Phase 2에서 scale dtype을
  artifact metadata에 명시한다.

## Storage estimate (측정값)

`scripts/check_packed_ternary.py`의 `storage_report` 기준.

```text
512 x 2048 linear, group_size 64 (n_groups 32), fp16 scales:
  trit    : 236.8 KB   (8.65x vs fp16)
  two_bit : 288.1 KB   (7.11x vs fp16)
  int8    : 1024  KB   (2.00x)
  fp16    : 2048  KB   (1.00x)
  ideal b1.58 (scale 제외) 와 trit 차이: 1.600 vs 1.585 bits/elem
```

scale은 numel 대비 `n_groups/in_features` 비율로 작아 압축률을 거의 깎지 않는다
(위 예: scale 32KB vs code 256KB).

## TC matrix

| ID | 영역 | 검증 | Pass 기준 | 상태 |
| --- | --- | --- | --- | --- |
| PACK-001 | Domain | unpack 결과가 {-1,0,+1} | subset | PASS |
| PACK-002 | Round trip | unpack(pack(T)) == T (양 scheme) | 정확히 동일 | PASS |
| PACK-003 | Dense match | to_dense() == conversion.S1 alpha*T | max_err < 1e-6 | PASS (0.0) |
| PACK-004 | SSTE export | pack(ScaledBitLinear).to_dense() == forward weight | max_err < 1e-6 | PASS (7e-9) |
| PACK-005 | Save/Load | load(save(x)).to_dense() == x.to_dense() | 동일 | PASS |
| PACK-006 | Storage | trit < two_bit < int8 < fp16, trit~1.585 | ordering + near-bound | PASS |
| PACK-101 | Model logit equality | pack -> unpack model logits == S1-converted model logits | max_err < 1e-5 | PASS (0.0) |
| PACK-102 | Model artifact save/load | load(save(artifact)) logits == S1-converted model logits | max_err < 1e-5 | PASS (0.0) |
| PACK-103 | Whole-model storage | packed target linears + fp16 others < fp16 model | ratio > 1 | PASS (3.78x) |
| PACK-201 | Packed linear forward | `PackedTernaryLinear(x) == F.linear(x, S1 alpha*T)` | max_err < 1e-5 | PASS (0.0) |
| PACK-202 | Packed module model logits | target linears swapped to packed modules == S1-converted model | max_err < 1e-5 | PASS (0.0) |
| PACK-203 | No dense weight param | packed modules hold uint8 codes and no float `[out,in]` weight parameter | all target modules | PASS (14) |
| PACK-204 | Packed runtime state round-trip | save/load packed-module state keeps logits | max_err < 1e-5 | PASS (0.0) |

검증:

```bash
.venv/bin/python -m py_compile bitnet_llama/packing.py
.venv/bin/python scripts/check_packed_ternary.py --json-out reports/packed_ternary_tc.json --strict
.venv/bin/python scripts/check_packed_model.py --json-out reports/packed_model_tc.json --strict
.venv/bin/python scripts/check_packed_runtime.py --json-out reports/packed_runtime_tc.json --strict
```

## Implementation phases

### Phase 1 (완료): format + CPU reference pack/unpack

- `bitnet_llama/packing.py`: two_bit/trit pack·unpack, groupwise alpha, `PackedTernaryWeight`,
  `pack_scaled_bitlinear`, `storage_report`, save/load.
- `scripts/check_packed_ternary.py`: PACK-001..006.

### Phase 2 (완료): 모델 단위 export/import

- `LlamaForCausalLM`의 target linear들을 일괄 pack -> 단일 artifact로 저장.
- 제외 layer(embedding/lm_head/norm)는 fp16 유지.
- artifact 전체 storage vs fp16 baseline 리포트.
- round-trip 후 model forward logit 동일성(TC).

결과:

```text
PACK-101 logit equality  : max_logit_err=0.00e+00
PACK-102 save/load model : max_logit_err=0.00e+00
PACK-103 model storage   : 14 layers packed, 3.78x vs fp16
                            203.3 KB packed artifact target + fp16 others
                            769.2 KB fp16 baseline
                            target fraction 0.83
```

이 숫자는 layer-only `8.65x`보다 낮다. embedding/lm_head/norm/bias를 fp16으로
남긴 모델 전체 기준이기 때문이다. 이 때문에 이후 pretrained 모델에서도 layer-only
압축률이 아니라 model-specific whole artifact storage를 항상 같이 보고한다.

### Phase 3 (완료): `PackedTernaryLinear` reference runtime PoC

- `PackedTernaryWeight`를 들고 있는 `nn.Module`을 만든다.
- forward에서는 packed code를 on-the-fly unpack -> dense `alpha*T` -> `F.linear`로 계산한다.
- 속도 최적화가 아니라 runtime wiring 정확성만 본다.
- packed-module model과 Phase 2 unpacked S1 model의 logit 동일성을 확인한다.
- 이 단계가 통과해야 kernel PoC가 "저장 format"이 아니라 "실제 추론 모듈" 위에 올라간다.

결과:

```text
PACK-201 layer forward       : max_err=0.00e+00
PACK-202 model logits        : 14 packed modules, max_err=0.00e+00
PACK-203 no dense weight     : uint8 codes, no float weight parameter
PACK-204 state round-trip    : max_err=0.00e+00
target linear storage        : 74.0 KB packed vs 640.0 KB fp16, 8.65x
```

정직한 한계:

```text
PackedTernaryLinear is a reference runtime module, not a speed kernel.
It stores packed bytes, but forward still materializes dense alpha*T before
calling F.linear. Storage/load memory is reduced; compute-time peak memory and
latency are not solved yet.
```

### Phase 4: kernel PoC 판단

- CPU에서 dense `[out,in]` weight materialization을 피하는 unpack-free 또는
  fused dequant matmul reference PoC.
- correctness target: Phase 3 `PackedTernaryLinear`와 logit 동일.
- measurement target: dense materialize path보다 lower peak intermediate memory를 보이는지 확인.
- 그 뒤 Metal/CUDA/bitnet.cpp 타겟 결정.

## Watch item: KL-to-fp16

real-text에서 scaled-STE는 CE/PPL/accuracy가 projected-QAT보다 좋지만 KL-to-fp16은
약간 높았다(예: 0.18 vs 0.11). 이 format 자체는 `alpha*T`를 정확히 복원하므로 KL을
바꾸지 않는다. KL은 export 후 logit 비교(Phase 2~3)에서 fp16 dense 대비 회귀 신호로만
추적한다. CE/PPL가 더 좋은 한 pause 신호가 아니다.

## Decision after this phase

format/TC가 고정되고 `PackedTernaryLinear` reference runtime도 logit 동일성을 통과했다. 다음은:

1. CPU fused/dequant matmul reference로 dense weight materialization 제거 가능성 확인
2. peak intermediate memory와 latency를 정직하게 측정
3. kernel 타겟(우선 CPU, 그 다음 Metal/CUDA/bitnet.cpp) 스코핑
4. TurboQuant KV cache는 weight format이 런타임에서 검증된 뒤 별도 축으로
