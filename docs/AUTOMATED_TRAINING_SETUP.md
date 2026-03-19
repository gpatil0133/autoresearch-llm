# Automated Training Setup

This document provides a detailed guide to set up and run the automated training loop for survey text analytics.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the following files are configured:
   - `prepare.py`: Data preparation and schema definitions.
   - `model_selector.py`: Model registry and selection logic.
   - `train.py`: Training script with early stopping.
   - `inference.py`: Structured inference logic.
   - `eval.py`: Evaluation and benchmarking.

## Automated Training Loop

1. **Prepare Data**:
   - Place labeled survey data in `./data/surveys.csv`.
   - Run `prepare.py` to preprocess data.

2. **Select Model**:
   - Update `model_selector.py` to include desired models.
   - Supported models: Qwen, Llama, Mistral.

3. **Train**:
   - Execute `train.py` with the selected model.
   - Example:
     ```bash
     python train.py --model Mistral-7B
     ```

4. **Inference**:
   - Run `inference.py` to generate predictions.
   - Example:
     ```bash
     python inference.py --model Mistral-7B
     ```

5. **Evaluate**:
   - Use `eval.py` to compute metrics and update results.
   - Example:
     ```bash
     python eval.py
     ```

6. **Benchmark**:
   - Append results to `results.tsv`.
   - Update `leaderboard.json`.

## Individual Steps

- **Data Preparation**:
  ```bash
  python prepare.py
  ```
- **Training**:
  ```bash
  python train.py --model Mistral-3B
  ```
- **Inference**:
  ```bash
  python inference.py --model Mistral-3B
  ```
- **Evaluation**:
  ```bash
  python eval.py
  ```

## Notes

- Ensure VRAM requirements are met for larger models.
- Use `--help` with any script for additional options.
