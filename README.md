# LoRA Fine-tuning with PEFT for Qwen Model on MetaMathQA

This project implements LoRA (Low-Rank Adaptation) fine-tuning using the PEFT library to fine-tune the Qwen model on the MetaMathQA dataset for improved mathematical reasoning.

## Overview

- **Model**: Qwen3-0.6B-Base
- **Dataset**: meta-math/MetaMathQA
- **Training Samples**: 1,000 question-answer pairs
- **Evaluation Samples**: 200 question-answer pairs
- **Method**: LoRA fine-tuning with PEFT library

## Features

1. **Proper Answer Parsing**: Extracts answers in the format "The answer is: {final_answer}"
2. **Comprehensive Evaluation**: Compares original vs fine-tuned model performance
3. **Efficient Training**: Uses LoRA to fine-tune only a small percentage of parameters
4. **Detailed Logging**: Saves evaluation results and comparison metrics

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- transformers>=4.35.0
- peft>=0.7.0
- datasets>=2.14.0
- accelerate>=0.24.0
- bitsandbytes>=0.41.0
- tqdm>=4.65.0

### Step 2: Verify CUDA (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Option 1: Evaluate Original Model First (Recommended)

Before training, you can evaluate the baseline performance of the original Qwen3-0.6B model:

```bash
# Using the shell script
./run_original_eval.sh

# Or directly with Python
python evaluate_original_model.py
```

This will:
1. Load 200 evaluation samples from MetaMathQA
2. Evaluate the original Qwen3-0.6B-Base model
3. Save results to `original_model_evaluation.json`
4. Display accuracy and sample predictions

**Why do this first?**
- Get baseline accuracy quickly (~10-20 minutes on CPU)
- Verify your setup works before committing to training
- Understand what improvement to expect from fine-tuning

### Option 2: Run the Complete Pipeline

```bash
# Using the shell script
./run_training.sh

# Or directly with Python
python lora_finetuning.py
```

This script will:
1. Load 1,200 samples from MetaMathQA (1,000 for training, 200 for evaluation)
2. Evaluate the original Qwen model on the evaluation set
3. Fine-tune the model using LoRA with PEFT (~1-3 hours)
4. Evaluate the fine-tuned model on the same evaluation set
5. Compare and report the performance improvements

## LoRA Configuration

The script uses the following LoRA parameters:

- **Rank (r)**: 16 - Controls the dimensionality of low-rank matrices
- **Alpha**: 32 - Scaling factor for LoRA weights
- **Target Modules**: ["q_proj", "k_proj", "v_proj", "o_proj"] - Attention layers to fine-tune
- **Dropout**: 0.05 - Regularization to prevent overfitting
- **Task Type**: CAUSAL_LM - Causal language modeling task

## Training Configuration

- **Epochs**: 3
- **Batch Size**: 4
- **Gradient Accumulation Steps**: 4 (effective batch size: 16)
- **Learning Rate**: 2e-4
- **FP16**: Enabled if CUDA is available
- **Warmup Steps**: 50
- **Weight Decay**: 0.01

## Output Files

### From Original Model Evaluation (`evaluate_original_model.py`)

- **`original_model_evaluation.json`** - Baseline evaluation results including:
  - Overall accuracy
  - Per-sample results (query, ground truth, prediction, correctness)
  - Model and dataset information

### From Complete Training Pipeline (`lora_finetuning.py`)

1. **`lora_finetuned_model/`** - Directory containing the fine-tuned model with LoRA adapters
2. **`original_model_results.json`** - Detailed evaluation results for the original model
3. **`finetuned_model_results.json`** - Detailed evaluation results for the fine-tuned model
4. **`comparison_results.json`** - Summary comparison showing accuracy improvement

## Answer Parsing

The script properly handles the answer format "The answer is: {final_answer}" by:

1. Extracting the final answer using regex patterns
2. Normalizing answers (removing commas, dollar signs, etc.)
3. Comparing normalized answers for accuracy calculation

## Expected Results

After fine-tuning, you should see:
- Improved accuracy on mathematical reasoning tasks
- The fine-tuned model better follows the required answer format
- Detailed metrics showing the improvement over the original model

## Model Information

**Note**: The script uses `Qwen/Qwen3-0.6B-Base` as the base model. If you want to use a different Qwen model variant, modify the `MODEL_NAME` variable in `lora_finetuning.py`:

```python
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"  # Change this to your desired model
```

Available Qwen3 models include:
- `Qwen/Qwen3-0.6B-Base` (Base model - recommended for fine-tuning)
- `Qwen/Qwen3-0.6B` (Instruct/chat version)
- `Qwen/Qwen3-1.7B-Base`
- `Qwen/Qwen3-4B-Base`

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:
1. Reduce batch size in `training_args.per_device_train_batch_size`
2. Increase gradient accumulation steps
3. Reduce max sequence length in `preprocess_function`

### Slow Training

If training is slow on CPU:
1. Consider using a smaller model
2. Reduce the number of training samples
3. Use a machine with GPU support

### Import Errors

Make sure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Technical Details

### LoRA (Low-Rank Adaptation)

LoRA fine-tunes large language models efficiently by:
- Freezing the original model weights
- Adding trainable low-rank matrices to specific layers
- Significantly reducing the number of trainable parameters (typically <1%)
- Maintaining or improving model performance

### PEFT (Parameter-Efficient Fine-Tuning)

The PEFT library by HuggingFace provides:
- Easy integration with Transformers
- Multiple fine-tuning methods (LoRA, Prefix Tuning, etc.)
- Efficient memory usage
- Simple model saving and loading

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MetaMathQA Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)
- [Qwen Models](https://huggingface.co/Qwen)
