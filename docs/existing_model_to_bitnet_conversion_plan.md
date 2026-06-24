# Existing Model to BitNet Conversion Plan

문서 위치: [Index](./index.md) -> 기존 dense model을 BitNet-style ternary로 변환하는 전체 ladder

관련 문서:

- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)

## 목적

이 문서는 기존 full-precision LLM checkpoint를 teacher distillation 없이 BitNet-style ternary 모델로 변환할 수 있는지 검증하기 위한 계획서다. 목표는 "새로 BitNet을 pretrain"하거나 "teacher logits로 증류"하는 것이 아니라, 일반 quantization처럼 기존 모델을 후처리, calibration, reconstruction, 짧은 후학습으로 1.58-bit weight domain에 가깝게 내리는 것이다.

핵심 질문:

- 기존 LLaMA/Qwen 계열 모델의 `nn.Linear` weight를 `{ -1, 0, +1 }` 기반 표현으로 변환할 수 있는가?
- 단순 PTQ-style ternarization만으로 어느 정도 품질이 유지되는가?
- teacher distillation 없이 calibration text, reconstruction, LM loss 기반 후학습만으로 회복 가능한가?
- 어떤 레이어, 어떤 projection, 어떤 scale granularity가 실패 지점인가?

참고할 최신 방향:

- BitNet b1.58: https://arxiv.org/abs/2402.17764
- Extra RMSNorm fine-tuning to 1.58 bits: https://arxiv.org/abs/2505.08823
- ParetoQ extremely low-bit quantization: https://arxiv.org/abs/2502.02631
- PTQ to Trit-Planes: https://arxiv.org/abs/2509.16989
- BitNet Distillation, 비교 대상으로만 사용: https://arxiv.org/abs/2510.13998

## 비목표

- teacher logits, hidden states, attention maps를 쓰는 knowledge distillation은 하지 않는다.
- 처음부터 packed 1.58-bit storage나 custom kernel을 구현하지 않는다.
- conversion feasibility가 확인되기 전 bitnet.cpp/GGUF export를 목표로 삼지 않는다.
- TurboQuant KV cache 압축은 이 계획의 주 목표가 아니다.

## 방향 판단

이 프로젝트에는 TurboQuant보다 이 실험이 더 잘 맞는다. 현재 저장소의 핵심은 `nn.Linear`를 `BitLinear`로 교체하는 구조이므로, 기존 모델의 linear weight를 ternary domain으로 바꿔서 forward 품질이 얼마나 유지되는지 보는 데 바로 활용할 수 있다.

다만 "weight만 한 번에 ternarize하면 된다"는 가정은 위험하다. quantization에서도 PTQ, calibration, reconstruction, QAT/fine-tuning이 이어지는 연속선이 있으므로, 1.58-bit/ternary 변환도 다음 ladder를 순서대로 검정한다.

```text
S0: naive ternary PTQ
S1: scaled/groupwise ternary PTQ
S2: layerwise reconstruction calibration
S3: post-training no-teacher QAT with LM loss
S4: stabilization modules, e.g. extra RMSNorm
S5: structured ternary fallback, e.g. trit-plane style
```

S0/S1에서 품질이 무너지면 정상이다. 이 실험의 가치는 어느 수준의 후처리와 후학습부터 회복되는지, 어떤 projection이 병목인지, teacher 없이 가능한 최소 절차가 무엇인지 찾는 데 있다.

## 용어와 경계

이 계획에서 "기존 모델 변환"은 딱딱한 one-shot PTQ만 뜻하지 않는다. 다음은 모두 허용 범위다.

- weight-only PTQ: calibration 없이 weight 통계만 사용한다.
- calibration-aware PTQ: unlabeled text에서 activation 통계를 모아 scale/threshold를 고른다.
- reconstruction-based quantization: 각 layer의 입력 activation `X`를 사용해 `XW`와 `XW_approx` 차이를 줄인다.
- post-training QAT/fine-tuning: 기존 checkpoint에서 시작해 STE ternary layer를 켜고 일반 LM CE loss로 짧게 회복한다.
- stabilization adapter: Extra RMSNorm처럼 작은 구조 보강을 넣고 후학습한다.

명확히 제외하는 것은 knowledge distillation이다. 즉 full-precision teacher의 logits, hidden states, attention maps를 target으로 삼는 loss는 쓰지 않는다. 원본 checkpoint의 weight와 calibration activation을 분석하는 것은 허용한다.

## 변환 대상

1차 대상은 작은 causal LM이다.

- local tiny random LLaMA config
- TinyLlama/Qwen small class 모델, 환경이 허용될 때
- 이 저장소의 `bitllama-110M-config`

큰 모델은 conversion pipeline이 안정화된 뒤 검증한다.

## 변환 대상 레이어 정책

우선 적용:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

초기 제외:

- token embedding
- final `lm_head`
- normalization layers

이유:

- BitNet 계열도 주로 linear projection 대체를 핵심으로 둔다.
- `lm_head`와 embedding은 vocab dimension이 커서 품질 충격과 저장 정책을 별도로 봐야 한다.

## Weight 표현

초기 reference 표현:

```text
W_fp      : original full precision weight
T         : ternary code in {-1, 0, +1}
alpha     : scale, per-tensor / per-output-channel / per-group
W_approx  : alpha * T
```

후보:

1. per-tensor scale
2. per-output-channel scale
3. groupwise scale along output dimension
4. groupwise scale along input blocks

기본 후보는 per-output-channel scale이다. `nn.Linear` weight shape이 `[out_features, in_features]`이므로 output channel별 dynamic range 차이를 흡수하기 쉽다.

## 변환 알고리즘 후보

### S0: naive ternary PTQ

목표:

- calibration 없이 weight만 보고 ternarize한다.

공식 초안:

```text
threshold = lambda * mean(abs(W))
T = sign(W) if abs(W) > threshold else 0
alpha = mean(abs(W[abs(W) > threshold]))
W_approx = alpha * T
```

검정 목적:

- 가장 싼 변환의 하한선을 측정한다.

### S1: scaled/groupwise ternary PTQ

목표:

- scale granularity와 threshold 탐색만으로 layer reconstruction error를 낮춘다.

후보 탐색:

- `lambda in {0.3, 0.5, 0.7, 1.0}`
- per-tensor, per-channel, groupwise scale
- sparsity ratio 기록

검정 목적:

- 단순 PTQ가 완전히 무의미한지, projection별로 살릴 수 있는 구간이 있는지 확인한다.

### S2: layerwise reconstruction calibration

목표:

- teacher model 출력이 아니라 calibration activation `X`를 사용해 `XW`와 `XW_approx` 차이를 최소화한다.

허용 데이터:

- unlabeled text
- tokenizer로 만든 input ids
- 원본 model forward에서 hook으로 캡처한 각 linear input activation

금지:

- teacher logits loss
- teacher hidden-state matching loss
- teacher attention map loss

목적 함수:

```text
min alpha, threshold || X W_fp^T - X (alpha * T)^T ||_2
```

검정 목적:

- "weight MSE가 낮음"이 아니라 "layer output error가 낮음"을 기준으로 conversion을 개선한다.

### S3: post-training no-teacher QAT with LM loss

목표:

- 기존 checkpoint에서 시작해, ternary STE를 켠 상태로 일반 next-token LM loss만으로 짧게 회복한다.

허용:

- training text의 next-token labels
- CE loss
- STE
- gradual layerwise quantization schedule

금지:

- full-precision teacher logits
- KL distillation
- intermediate distillation

검정 목적:

- 이건 one-shot PTQ는 아니지만, quantization에서 흔히 쓰는 후학습/QAT 변환에 해당한다.
- 핵심 제약은 teacher-free이며, 일반 LM objective로 기존 checkpoint를 회복한다는 점이다.

### S4: Extra RMSNorm stabilization

목표:

- every linear projection 앞에 RMSNorm 또는 scale-normalization adapter를 넣어 ternary fine-tuning 안정성을 확인한다.

주의:

- architecture가 바뀌므로 S0-S3와 별도 축으로 비교한다.
- adapter 추가 파라미터 수를 리포트해야 한다.

검정 목적:

- 기존 모델을 1.58-bit로 내릴 때 normalization만으로 회복 폭이 커지는지 확인한다.

### S5: structured ternary fallback

목표:

- plain `{ -1, 0, +1 }` single plane이 실패할 경우, PTQTP류 structured ternary decomposition을 fallback으로 검토한다.

예:

```text
W_approx = alpha_1 * T_1 + alpha_2 * T_2
T_i in {-1, 0, +1}
```

주의:

- 이 경우 1.58-bit single plane BitNet은 아니다.
- "BitNet-like ternary quantization"으로 따로 표기한다.

## Core TC Matrix

| ID | 영역 | 검증 내용 | 방법 | Pass 기준 |
| --- | --- | --- | --- | --- |
| CONV-001 | Mapping | 원본 Linear와 변환 BitLinear의 shape가 동일하다 | state_dict 변환 후 shape 비교 | 모든 target layer shape 동일 |
| CONV-002 | Exclusion | embedding/lm_head/norm은 초기 변환에서 제외된다 | converted state_dict key 확인 | 제외 key가 변환되지 않음 |
| TERN-001 | Domain | ternary code가 {-1,0,+1}만 가진다 | unique value 검사 | subset of {-1,0,+1} |
| TERN-002 | Sparsity | zero ratio가 기록된다 | layer별 count | report 생성 |
| SCALE-001 | Scale | scale이 finite이고 shape policy와 일치한다 | per-layer scale 검사 | NaN/Inf 없음 |
| PTQ-001 | Weight Error | ternary approximation이 zero baseline보다 낫다 | `MSE(W, W_approx)` vs `MSE(W, 0)` | 모든 변환 layer에서 개선 |
| PTQ-002 | Layer Error | `XW_approx`가 `XW`를 일정 수준 보존한다 | calibration activation으로 비교 | projection별 relative error 기록 |
| PTQ-003 | Search | threshold/grid search가 naive보다 악화되지 않는다 | S0 vs S1 비교 | avg layer output error 감소 또는 동일 |
| QAT-001 | STE | ternary layer의 gradient가 끊기지 않는다 | small CE loss backward | finite grad |
| QAT-002 | No Teacher | QAT loop가 teacher logits/hidden을 참조하지 않는다 | code path/static test | teacher loss/reference 없음 |
| QAT-003 | Schedule | gradual quantization이 layer order를 지킨다 | schedule trace | 지정 순서와 일치 |
| SSTE-001 | Scaled STE | `ScaledBitLinear` forward weight가 S1 `alpha*T`와 일치한다 | `scripts/check_scaled_bitlinear.py` | max abs diff <= 1e-6 |
| SSTE-002 | Scaled STE | latent full-precision weight로 gradient가 흐른다 | MSE backward smoke | finite non-zero grad |
| SSTE-003 | Scaled STE | activation fake quant option이 NaN을 만들지 않는다 | 8-bit activation smoke | finite output |
| EVAL-001 | Disabled Equivalence | conversion off는 baseline과 동일하다 | same model forward | logits max diff == 0 |
| EVAL-002 | PTQ Perplexity | S0/S1 변환 후 PPL을 측정한다 | Wikitext subset | pass/fail 대신 baseline delta 기록 |
| EVAL-003 | Recovery | S2/S3가 S0/S1보다 PPL을 회복한다 | 동일 eval set 비교 | PPL 감소 |
| EVAL-004 | Generation Smoke | 변환 모델이 generate를 수행한다 | 짧은 prompt | NaN 없음, shape error 없음 |
| SER-001 | Save/Load | 변환 state를 저장/로드할 수 있다 | save/load round trip | logits finite, metadata 유지 |
| MEM-001 | Storage Estimate | 실제 packed 전이라도 이론 저장량을 계산한다 | ternary code + scale bytes 산출 | report 생성 |

## Escalation and Kill Criteria

아래 조건은 즉시 폐기 신호가 아니라, 먼저 더 강한 변환 단계로 올릴지 판단하는 기준이다. 같은 문제가 여러 schedule과 작은 budget sweep에서도 반복될 때만 중단하거나 fallback으로 보낸다.

- S1이 S0보다 layer output error를 줄이지 못하면 threshold/scale policy를 늘려본다.
- S2가 layer output error를 줄여도 PPL이 회복되지 않으면 layer order와 outlier projection을 분석한다.
- S3 post-training QAT가 짧은 budget에서 S1보다 PPL을 줄이지 못하면 learning rate, quantization schedule, unfrozen layer 범위를 sweep한다.
- STE gradient가 불안정하게 NaN/Inf를 만든다.
- 특정 projection, 예를 들어 `down_proj`나 `o_proj`, 하나가 전체 품질 붕괴를 지배하면 해당 projection만 delayed quantization 또는 higher-rank ternary fallback으로 보낸다.
- teacher-free 조건을 어기는 순간 이 실험 목적에서 벗어난다.

## 구현 단계

### Phase A: conversion reference

산출물:

- `bitnet_llama/conversion.py`
- `tests/test_bitnet_conversion.py`

내용:

- target linear key discovery
- ternary code generation
- scale computation
- converted state_dict 생성
- layer별 stats report

검증:

```bash
python3 -m py_compile bitnet_llama/conversion.py
pytest tests/test_bitnet_conversion.py
```

### Phase B: calibration harness

산출물:

- `bitnet_llama/calibration.py`
- `scripts/collect_calibration_stats.py`

내용:

- calibration text loading
- linear input activation hook
- layer output reconstruction error 측정
- threshold/scale policy search

검증:

```bash
pytest tests/test_bitnet_calibration.py
python scripts/collect_calibration_stats.py --model <model> --max-samples 32
```

### Phase C: post-training no-teacher QAT

산출물:

- `scripts/convert_to_bitnet_qat.py`
- `tests/test_no_teacher_qat.py`

내용:

- original checkpoint load
- BitLinear 교체
- STE ternary forward
- CE loss only fine-tuning
- gradual layerwise quantization schedule

검증:

```bash
pytest tests/test_no_teacher_qat.py
python scripts/convert_to_bitnet_qat.py --model <model> --max-steps 20 --smoke
```

### Phase D: eval and report

산출물:

- `scripts/evaluate_bitnet_conversion.py`
- `reports/conversion_<model>.json`

내용:

- baseline PPL
- S0/S1/S2/S3 PPL
- generation smoke
- layer error distribution
- theoretical storage estimate

검증:

```bash
python scripts/evaluate_bitnet_conversion.py --model <model> --eval wikitext --max-samples 128
```

## 실험 판정 기준

이 실험은 바로 성공/실패가 아니라 ladder별로 판단한다.

```text
S0/S1만으로 PPL delta가 작다:
  -> 기존 모델 PTQ-style BitNet 변환 가능성이 큼.

S0/S1은 망가지지만 S2가 회복한다:
  -> calibration-aware ternary conversion 가능성 있음.

S2도 부족하지만 S3가 회복한다:
  -> one-shot PTQ가 아니라 후학습 포함 quantization 변환이 현실적.

S3도 부족하고 distillation 논문만큼 회복 안 됨:
  -> teacher-free 변환은 더 긴 budget, stabilization, structured ternary fallback이 필요.

S4가 큰 폭으로 회복한다:
  -> Extra RMSNorm류 구조 변경이 핵심.

S5만 가능하다:
  -> single-plane BitNet 변환보다 structured ternary PTQ가 더 현실적.
```

## TurboQuant 문서와의 관계

TurboQuant는 KV cache 압축 문제다. 이 문서의 목표는 기존 model weight를 BitNet-style ternary domain으로 변환하는 것이다. 두 방향은 나중에 결합할 수 있지만, 현재 우선순위는 다음과 같다.

1. 기존 모델 weight를 ternary/BitNet으로 변환할 수 있는지 검증한다.
2. 변환 모델이 의미 있는 품질을 유지하면 inference memory 병목으로 KV cache 압축을 검토한다.
3. 그때 TurboQuant/RateQuant/HyperQuant 후보를 비교한다.
