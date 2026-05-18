# RLHF for Public Value Alignment in Government LLMs

Master's thesis codebase — applying **Reinforcement Learning from Human Feedback (RLHF)** to align LLM responses with transparency values in public administration.

**Domain:** Amsterdam parking permit queries  
**Base model:** Mistral-7B-Instruct-v0.3  
**Pipeline:** SFT → Preference Simulation (LLM-as-judge) → Reward Model Training → PPO (external)

## Project Structure

```
Master_Thesis_Github_clone/
├── data/                          # Q&A datasets, queries, validation CSV
├── src/
│   ├── config.py                  # Shared configuration and paths
│   ├── data_processing/
│   │   └── preprocess.py          # Tokenize Q&A with chat-template masking
│   ├── sft/
│   │   └── train.py               # LoRA SFT training (MPS/CUDA, optional QLoRA)
│   ├── inference/
│   │   └── run_inference.py       # Generate answers from fine-tuned model
│   ├── rlhf/
│   │   ├── generate_responses.py  # Produce diverse responses for preference data
│   │   ├── simulate_preferences.py # LLM-as-judge preference annotation
│   │   ├── combine_datasets.py    # Merge judge outputs and compute agreement
│   │   └── reward_model/
│   │       ├── dataset.py         # PreferenceDataset (chosen vs. rejected pairs)
│   │       ├── model.py           # RewardModel with Bradley-Terry loss
│   │       ├── train.py           # Reward model training loop
│   │       └── evaluate.py        # Qualitative checkpoint evaluation
│   ├── evaluation/
│   │   └── bar_plot.py            # Publication-quality SFT vs PPO bar chart
│   └── utils/
│       ├── io_utils.py            # JSON/JSONL helpers
│       └── model_utils.py         # Tokenizer/model loading, prompt formatting
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Pipeline Overview (Execution Order)

1. **Preprocess** — Tokenize the Q&A dataset with chat-template label masking
   ```bash
   python -m src.data_processing.preprocess
   ```

2. **SFT Training** — Fine-tune Mistral-7B with LoRA (unquantized, 8-bit, or 4-bit)
   ```bash
   python -m src.sft.train --quantization_mode 4bit
   ```

3. **Inference** — Run the SFT model against validation queries
   ```bash
   python -m src.inference.run_inference \
     --validation_csv data/validation_qa.csv \
     --adapter_path ./sft_results/mistral_lora_4bit/final_model_adapters
   ```

4. **Generate RLHF responses** — Produce diverse completions from the SFT model
   ```bash
   python -m src.rlhf.generate_responses \
     --query_file data/rlhf_queries_50.jsonl \
     --adapter_path ./sft_results/mistral_lora_4bit/final_model_adapters
   ```

5. **Simulate preferences** — Use GPT-4.1 or Claude Sonnet as judge
   ```bash
   python -m src.rlhf.simulate_preferences \
     --provider openai \
     --simulator_model gpt-4.1 \
     --sft_responses_file rlhf_responses_20250526.json
   ```

6. **Combine datasets** — Merge GPT-4.1 and Sonnet-4 preference files
   ```bash
   python -m src.rlhf.combine_datasets \
     --files gpt4.1=gpt_prefs.jsonl sonnet4=sonnet_prefs.jsonl
   ```

7. **Train reward model** — Bradley-Terry pairwise ranking on combined preferences
   ```bash
   python -m src.rlhf.reward_model.train \
     --preference_csv combined_preferences.csv
   ```

8. **Evaluate reward model** — Qualitative test of preference ordering
   ```bash
   python -m src.rlhf.reward_model.evaluate \
     --checkpoint_dir ./reward_model_output/best_reward_model
   ```

9. **Bar plot** — SFT vs PPO comparison with significance testing
   ```bash
   python -m src.evaluation.bar_plot evaluation_results.csv --out bar_plot.pdf
   ```

## Key Numbers

| Metric | Value |
|---|---|
| Q&A pairs created | 106 |
| Q&A pairs used for SFT | 50 |
| SFT model variants trained | 3 (unquantized, 8-bit, 4-bit) |
| RLHF queries | 50 |
| Preference pairs generated | 300 (150 GPT-4.1 + 150 Sonnet 4) |
| Cross-LLM agreement | 61.8% (κ = 0.242) |
| Best RM validation accuracy | 76.7% (epoch 2) |

## Requirements

- Python 3.11+
- PyTorch 2.0+
- `transformers`, `peft`, `datasets`, `bitsandbytes`
- `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`
- `anthropic`, `openai` (for preference simulation)
- `matplotlib`, `seaborn` (for visualisation)

Install with: `pip install -r requirements.txt`

## Notes

- **Device support:** Scripts auto-detect CUDA > MPS > CPU. Quantization (8-bit/4-bit) requires CUDA + `bitsandbytes`.
- **API keys:** Set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` environment variables for preference simulation.
- **PPO step** was run externally on RunPod and is not included in this repository.

## Repository History & Architecture

This codebase originated from a 2025 Master's thesis (MSc Artificial Intelligence, University of Applied Sciences) by Julian Adam. The original implementation was developed across May–June 2025 as a collection of standalone scripts and Jupyter notebooks under a single flat directory. The pipeline was executed on a combination of a MacBook M1 (for dataset preparation and local SFT prototyping) and RunPod GPU instances (for quantized training and reward model fitting).

In 2026 the repository was structurally refactored for production readiness and long-term maintainability. Key improvements included:

- **Modular package layout** — Scripts were reorganised into a conventional `src/` package with clear separation between data processing, SFT, inference, RLHF components, and evaluation.
- **Type annotations** — All public functions and classes received full type hints (`from __future__ import annotations`) to support static analysis and IDE tooling.
- **Professional docstrings** — Every module and public callable was documented with Google-style docstrings describing parameters, return values, and behavioural intent.
- **Shared configuration** — Hardcoded paths, hyperparameters, and model identifiers were centralised in `src/config.py`.
- **Deduplicated utilities** — Common patterns such as model loading, tokenizer setup, prompt formatting, and file I/O were extracted into reusable utility modules.
- **Consolidated entry points** — Redundant scripts (separate training files for quantized and unquantized SFT; separate preference simulation scripts for Anthropic and OpenAI) were merged into single parameterised entry points.
- **Removed dead code** — Disused notebooks, debugging artifacts, and abandoned experimental variants were excluded.

The refactoring preserved the original pipeline logic and hyperparameter defaults. Every command in the original workflow has an equivalent `python -m src.<subpackage>.<module>` invocation documented above.
