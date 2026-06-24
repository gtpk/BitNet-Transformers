# TurboQuant + BitNet Implementation Plan

문서 위치: [Index](./index.md) -> weight 변환 이후 KV cache 압축 확장 계획

관련 문서:

- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Existing Model to BitNet Conversion Plan](./existing_model_to_bitnet_conversion_plan.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)

## 목적

이 문서는 BitNet-Transformers에 TurboQuant 아이디어를 넣기 전, 구현 범위와 테스트 케이스를 먼저 고정하기 위한 계획서다. 목표는 BitNet의 ternary weight 구조를 유지하면서, 긴 문맥 추론에서 커지는 KV cache를 TurboQuant 방식으로 압축하는 것이다.

핵심 판단:

- BitNet b1.58 계열은 model weight 압축을 담당한다.
- TurboQuant는 static weight가 아니라 runtime KV cache 압축에 먼저 적용한다.
- 1차 구현은 정확성 검증 가능한 PyTorch reference path로 만들고, packed bit storage와 fused kernel은 이후 단계로 분리한다.

참고 자료:

- BitNet b1.58: https://arxiv.org/abs/2402.17764
- TurboQuant: https://arxiv.org/abs/2504.19874
- BitNet v2 / H-BitLinear: https://arxiv.org/abs/2504.18415
- Microsoft bitnet.cpp: https://github.com/microsoft/BitNet

## 비목표

- Phase 1에서 bitnet.cpp 수준의 CPU/GPU kernel 성능을 바로 달성하지 않는다.
- Phase 1에서 GGUF export를 완성하지 않는다.
- TurboQuant를 BitLinear weight quantizer로 사용하지 않는다.
- 기존 `BitLinear`의 ternary STE 학습 의미를 바꾸지 않는다.
- `transformers` 내부 파일 symlink 방식의 전체 구조 개편은 별도 modernization phase로 둔다.

## 설계 원칙

1. 기본값은 항상 비활성화한다.
   - `turboquant_enabled=False` 상태에서는 기존 모델과 bit-level 동일한 출력을 내야 한다.

2. K와 V를 다르게 다룬다.
   - K cache는 attention score `QK^T`에 직접 들어가므로 inner product 보존이 중요하다.
   - V cache는 softmax weight로 재구성되는 값이므로 reconstruction MSE가 중요하다.

3. RoPE 이후의 K를 압축한다.
   - autoregressive cache에는 position 정보가 반영된 key가 저장되어야 한다.
   - 따라서 `key_states`, `value_states` 생성 후 RoPE 적용이 끝난 다음 encode한다.

4. reference path와 optimized path를 분리한다.
   - reference path: 압축 cache를 decode해서 기존 attention 연산에 넣는다.
   - optimized path: compressed-domain dot product, packed storage, fused kernel을 추후 추가한다.

5. deterministic metadata를 저장한다.
   - random rotation seed, quantization bit width, dimension, group/tile shape, residual settings를 cache metadata로 들고 간다.

## 제안 파일 구조

```text
bitnet_llama/
  turboquant.py              # TurboQuant reference encoder/decoder
  turboquant_cache.py        # Quantized KV cache wrapper
  module.py                  # BitLinear 계열, 변경 최소화
  modeling_llama.py          # LlamaAttention integration point

tests/
  test_turboquant_math.py    # 수학적 불변조건
  test_turboquant_cache.py   # KV cache shape/concat/decode
  test_bitnet_regression.py  # 기존 BitLinear 회귀
  test_llama_turboquant.py   # 모델 통합 smoke/equivalence

scripts/
  benchmark_turboquant_kv.py # memory/tokens/sec/quality 측정
```

## Config 초안

`LlamaConfig` 또는 별도 runtime argument에 다음 필드를 둔다.

```python
turboquant_enabled: bool = False
turboquant_k_bits: float = 3.5
turboquant_v_bits: float = 3.5
turboquant_rotation: str = "random_hadamard"
turboquant_seed: int = 0
turboquant_tile_size: int = 128
turboquant_inner_product_correction: bool = True
turboquant_reference_decode: bool = True
```

초기 구현에서는 fractional bit packing을 하지 않고, `3.5` 같은 값은 quantizer profile 이름으로 해석한다. 실제 packed representation은 Phase 5에서 추가한다.

## 알고리즘 적용 위치

현재 `LlamaAttention.forward()` 흐름에서 다음 위치에 들어간다.

```text
hidden_states
  -> q_proj/k_proj/v_proj
  -> reshape heads
  -> apply_rotary_pos_emb(q, k)
  -> TurboQuant encode k/v when use_cache=True
  -> store TurboQuantizedKVCache
  -> reference_decode for attention
  -> attention matmul/softmax/value matmul
```

1차 prototype은 `use_cache=True`인 generation path를 대상으로 한다. training forward에서는 기본 비활성화한다.

## Core TC Matrix

아래 TC는 구현 전에 먼저 작성한다. 이 중 `MATH`, `BNT`, `KV`, `INT`는 알고리즘 근간을 지키는 필수 테스트다.

| ID | 영역 | 검증 내용 | 방법 | Pass 기준 |
| --- | --- | --- | --- | --- |
| TQ-MATH-001 | Rotation | random rotation이 norm과 dot product를 보존하는지 확인 | fp32 random tensor에 rotation 적용 전후 norm/dot 비교 | max relative error <= 1e-5 |
| TQ-MATH-002 | Determinism | 같은 seed와 shape는 같은 rotation/encoding을 만든다 | 동일 입력을 2회 encode | code, metadata, decoded tensor 동일 |
| TQ-MATH-003 | MSE | bit width가 증가하면 평균 reconstruction MSE가 악화되지 않는다 | 2.5, 3.5, 4.5 profile로 random tensor encode/decode | avg MSE 4.5 <= 3.5 <= 2.5 |
| TQ-MATH-004 | Inner Product | K용 product quantizer가 내적 bias를 줄인다 | random q,k pair에서 original dot과 estimated dot 비교 | baseline scalar quant보다 mean bias 감소 |
| TQ-MATH-005 | Tail Dimensions | head_dim이 tile size로 나누어떨어지지 않아도 동작한다 | dim 64, 80, 96, 128, 160 입력 | shape 보존, NaN 없음 |
| TQ-MATH-006 | Residual QJL | residual correction을 켰을 때 inner product error가 감소한다 | correction on/off 비교 | 평균 absolute error 감소 |
| BNT-001 | BitLinear | ternary weight domain이 {-1, 0, 1}을 벗어나지 않는다 | `ternarize_weights_groupwise()` 결과 unique 확인 | unique subset of {-1, 0, 1} |
| BNT-002 | STE | BitLinear weight gradient가 끊기지 않는다 | 작은 loss backward | `weight.grad is not None`, finite |
| BNT-003 | Regression | TurboQuant 비활성화 시 기존 BitLinear 동작에 영향이 없다 | module import 및 forward smoke | 기존 TC와 동일 |
| KV-001 | Cache Shape | encoded cache가 batch/head/seq/head_dim 정보를 잃지 않는다 | 다양한 batch, num_heads, seq_len | decode shape 원본과 동일 |
| KV-002 | Cache Concat | incremental decoding에서 과거 cache와 새 token cache가 올바르게 concat된다 | seq를 token-by-token encode | full encode와 decode shape 및 순서 동일 |
| KV-003 | RoPE Ordering | RoPE 적용 전 key가 cache에 저장되지 않는다 | known position id로 key 비교 hook | cached key는 RoPE 후 key와 일치 |
| KV-004 | Device/Dtype | fp16/bf16/fp32와 CPU/GPU device metadata를 보존한다 | dtype/device별 encode/decode | output dtype 정책 일관, device mismatch 없음 |
| KV-005 | Memory | reference metadata overhead를 포함해 cache storage가 fp16보다 작아질 수 있는지 추적한다 | `numel * element_size + metadata` 계산 | Phase 2에서는 측정값 리포트, Phase 5에서 fp16 대비 감소 필수 |
| INT-001 | Disabled Equivalence | `turboquant_enabled=False`는 baseline과 정확히 같다 | same seed model forward 비교 | logits max abs diff == 0 |
| INT-002 | Reference Decode | `turboquant_enabled=True` reference path가 실행된다 | tiny config generation smoke | exception 없음, logits finite |
| INT-003 | Past KV API | Hugging Face legacy tuple cache와 호환된다 | `generate(max_new_tokens=...)` smoke | shape/API error 없음 |
| INT-004 | Attention Mask | padding/causal mask가 cache quantization과 충돌하지 않는다 | mask 포함 입력 | logits finite, no shape error |
| SER-001 | Serialization | config 저장/로드 후 TurboQuant 설정이 유지된다 | `save_pretrained/load_config` | field values identical |
| BENCH-001 | Quality Smoke | 짧은 eval에서 perplexity delta를 추적한다 | Wikitext subset | 3.5 profile delta <= 2% 목표 |
| BENCH-002 | Long Context | long-context retrieval smoke를 추적한다 | needle-in-a-haystack mini | retrieval success 유지 |
| BENCH-003 | Throughput | tokens/sec와 cache memory를 기록한다 | generation benchmark | baseline 대비 수치 리포트 생성 |

## 구현 단계

### Phase 0: 문서와 TC 고정

산출물:

- 이 문서
- TC skeleton 또는 pytest 파일 목록
- README 링크

완료 기준:

- 구현 전 테스트 기준이 명시되어 있다.
- pass/fail 기준이 문서화되어 있다.

### Phase 1: TurboQuant math reference

산출물:

- `bitnet_llama/turboquant.py`
- `tests/test_turboquant_math.py`

구현 내용:

- deterministic random rotation
- scalar quantizer profile
- MSE encode/decode
- product encode/decode
- optional residual QJL correction

완료 기준:

- `TQ-MATH-001`부터 `TQ-MATH-006`까지 통과
- torch CPU에서 동작
- no CUDA dependency

검증 명령:

```bash
python3 -m py_compile bitnet_llama/turboquant.py
pytest tests/test_turboquant_math.py
```

### Phase 2: Quantized KV cache wrapper

산출물:

- `bitnet_llama/turboquant_cache.py`
- `tests/test_turboquant_cache.py`

구현 내용:

- `TurboQuantizedKVCache`
- `encode_key`, `encode_value`, `decode_key`, `decode_value`
- cache concat
- metadata validation

완료 기준:

- `KV-001`부터 `KV-005`까지 통과
- reference decode path로 기존 attention에 넣을 수 있는 tensor를 반환

검증 명령:

```bash
python3 -m py_compile bitnet_llama/turboquant_cache.py
pytest tests/test_turboquant_cache.py
```

### Phase 3: LlamaAttention integration

산출물:

- `modeling_llama.py` integration patch
- `tests/test_llama_turboquant.py`

구현 내용:

- config flag 추가
- `use_cache=True` path에서만 encode
- disabled path의 exact equivalence 보장
- reference decode로 attention 수행

완료 기준:

- `INT-001`부터 `INT-004`까지 통과
- `turboquant_enabled=False`에서 기존 출력과 완전 동일
- tiny config generation smoke 통과

검증 명령:

```bash
python3 -m py_compile bitnet_llama/modeling_llama.py
pytest tests/test_llama_turboquant.py
```

### Phase 4: Quality and benchmark harness

산출물:

- `scripts/benchmark_turboquant_kv.py`
- benchmark result template

구현 내용:

- cache memory 계산
- tokens/sec 측정
- Wikitext subset perplexity
- needle-in-a-haystack mini

완료 기준:

- `BENCH-001`부터 `BENCH-003`까지 리포트 생성
- 품질/메모리/속도 trade-off를 동일 형식으로 비교 가능

검증 명령:

```bash
python scripts/benchmark_turboquant_kv.py --profile smoke
python scripts/benchmark_turboquant_kv.py --profile long-context
```

### Phase 5: Packed representation and optimized kernels

산출물:

- packed cache format
- optional C++/CUDA extension or bitnet.cpp bridge
- export notes

구현 내용:

- fractional bit profile의 실제 packing
- metadata overhead 축소
- compressed-domain or fused-decode attention
- bitnet.cpp/GGUF 호환성 조사

완료 기준:

- fp16 KV cache 대비 실제 storage 감소
- reference path와 optimized path의 numerical agreement 측정
- long-context benchmark에서 memory reduction 확인

## 중단 기준

다음 조건이 발생하면 다음 phase로 넘어가지 않는다.

- disabled equivalence가 깨진다.
- K cache product quantization이 scalar baseline보다 attention score bias를 줄이지 못한다.
- cache concat이 token 순서를 보존하지 못한다.
- RoPE 적용 전 key가 저장된다.
- BitLinear STE gradient가 끊긴다.
- reference implementation에서 NaN/Inf가 발생한다.

## 리뷰 체크리스트

- [ ] TurboQuant가 BitLinear weight 의미를 바꾸지 않는다.
- [ ] K와 V quantization 목표가 분리되어 있다.
- [ ] seed와 metadata가 재현 가능성을 보장한다.
- [ ] disabled path exact equivalence가 있다.
- [ ] 모든 TC가 독립적으로 실행 가능하다.
- [ ] benchmark가 품질, 메모리, 속도를 함께 기록한다.
- [ ] packed/kernel 최적화가 reference correctness 위에 올라간다.
