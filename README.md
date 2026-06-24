# 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of "BitNet: Scaling 1-bit Transformers for Large Language Models" in pytorch with Llama(2) Architecture

![BitNet Architecture](./static/bitnet-arch.png)

![BitNet](./static/bitnet.png)

- Paper Link: https://arxiv.org/pdf/2310.11453.pdf

## Modernization Notes

Start here:

- [Research index and reading path](./docs/index.md)

Core documents:

- [Memory-traffic-first BitNet plan](./docs/memory_traffic_first_plan.md)
- [Existing model to BitNet conversion plan](./docs/existing_model_to_bitnet_conversion_plan.md)
- [Scaled-STE BitLinear experiment](./docs/scaled_ste_bitlinear_experiment.md)
- [Evolutionary low-resource LLM arena plan](./docs/evolutionary_llm_arena_plan.md)
- [Colab arena runbook](./docs/colab_arena_runbook.md)
- [Colab validation summary](./docs/colab_validation_summary.md)
- [Real tiny text validation plan](./docs/real_tiny_text_validation_plan.md)
- [Research signal note](./docs/research_signal_note.md)
- [TurboQuant + BitNet KV-cache plan](./docs/turboquant_bitnet_implementation_plan.md)

## Low-Resource Experiment Runners

```bash
.venv/bin/python scripts/estimate_memory_traffic.py
.venv/bin/python scripts/run_arena_feasibility.py --strict
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_smoke_200.json --strict
.venv/bin/python scripts/check_scaled_bitlinear.py --json-out reports/scaled_bitlinear_tc.json
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_scaled_ste_smoke.json --strict
.venv/bin/python scripts/run_tiny_real_arena.py --data-mode text --text-path data/tiny_corpus.txt --train-steps 40 --qat-steps 12 --ste-qat-steps 12 --scaled-ste-steps 12 --seq-len 64 --batch-size 8 --eval-batch-size 16 --json-out reports/tiny_real_text_fixture_smoke.json
```

## Prepare Dev env

```bash
# Clone this repo
git clone https://github.com/beomi/bitnet-transformers
cd bitnet-transformers

# Install requirements
pip install -r clm_requirements.txt

# Clone transformers repo
git clone https://github.com/huggingface/transformers
pip install -e transformers

# Update Llama(2) model
rm ./transformers/src/transformers/models/llama/modeling_llama.py
ln -s $(pwd)/bitnet_llama/modeling_llama.py ./transformers/src/transformers/models/llama/modeling_llama.py
```

We'll overwrite `bitnet_llama/modeling_llama.py` into `transformers`. Since the file is linked, any changes made to the file will be reflected in the `transformers` repo.

## Train Wikitext-103

![Train Loss Graph when train BitLLAMA using Wikitext-103](./static/W&B_Chart_2023.10.20_wikitext.png)

> You can track metrics via wandb

```bash
./train_wikitext.sh
```

## GPU Mem Usage Comparison

**Train Config**

- Batch size: 1
- Gradient accumulation: 1
- Seq length: 2048
- Model: `LLamaForCausalLM` with `BitLinear` layer
- Model size: 47,452,672 (47.5M)

**Original LLAMA - 16bit**

- Uses **250MB** GPU memory for Model weights

**BitLLAMA - Mixed 16bit**

- Uses **200MB** GPU memory for Model weights
- Use bf16(or fp16) to store model weights
- Use int8 to store `-1`/`1` 1-bit weights
- Use more memory when training than original LLAMA: It saves 1-bit weight and 16bit weight together

**BitLLAMA - 8bit**

- Uses **100MB** GPU memory for Model weights
- Use bf16(or fp16) on-the-fly when needed
- Use 8bit to save 1-bit BitLinear weight & other weights

**BitLLAMA - 1bit**

- Use bf16(or fp16) on-the-fly when needed
- Use 1bit to save 1-bit weight

```bash
TBD
```

## Todo

- [x] Add `BitLinear` layer
- [x] Add `LLamaForCausalLM` model with `BitLinear` layer
    - [x] Update `.save_pretrained` method (for 1-bit weight saving)
- [x] Add sample code for LM training
- [ ] Update `BitLinear` layer to use 1-bit weight
    - [ ] Use uint8 instead of bfloat16
    - [ ] Use custom cuda kernel for 1-bit weight
