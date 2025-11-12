# LoRA Fine-tuning for Qwen3-0.6B on MetaMathQA

Fine-tuning Qwen3-0.6B-Base model using LoRA (Low-Rank Adaptation) for enhanced mathematical reasoning on the MetaMathQA dataset.

## 📊 Results Summary

Training configuration: 1,000 training samples, 100 validation samples, 100 test samples (all distinct, no overlap)

| Metric | Original Model | Fine-tuned Model | Improvement |
|--------|----------------|------------------|-------------|
| **Accuracy** | 48.00% | 59.00% | **+11.00%** |
| **Relative Gain** | - | - | **+22.9%** |
| **Trainable Params** | 0 | 4.6M (0.76%) | - |

**Key Achievement**: Significant accuracy improvement with minimal parameter updates.

**Name**: Qixuan Wang

**Hugging Face Fine-tuned Model Repo**: https://huggingface.co/kevinwang676/Qwen-0.6B-LoRA

**Results of the Original Model**: [original_model_results.json](./results/original_model_results.json)

**Results of the Fine-tuned Model**: [finetuned_model_results.json](./results/finetuned_model_results.json)

**LoRA Fine-tuning Script**: [lora_finetuning.py](./lora_finetuning.py)

**Unit Test Script**: [unit_test.py](./unit_test.py)

## 🐳 Docker Setup

### Build Docker Image

```bash
# Build locally
docker build -t lora-finetuning:latest .

# Or use helper script
./build_docker.sh
```

**Image Details**:
- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Size**: ~8GB (includes CUDA runtime and dependencies)
- **GPU Support**: NVIDIA CUDA 12.1+

### Run with Docker

**Unit Test**:
```bash
docker run --rm lora-finetuning:latest
```

**Full Training (GPU)**:
```bash
docker run --gpus all --rm \
  -v $(pwd)/output:/workspace \
  lora-finetuning:latest \
  python3 lora_finetuning.py
```

**Full Training (CPU)**:
```bash
docker run --rm \
  -v $(pwd)/output:/workspace \
  lora-finetuning:latest \
  python3 lora_finetuning.py
```

### Push to Docker Hub (Optional)

```bash
# Tag with your username
docker tag lora-finetuning:latest YOUR_USERNAME/lora-finetuning:latest

# Push to Docker Hub
docker push YOUR_USERNAME/lora-finetuning:latest

# Others can then pull and run
docker pull YOUR_USERNAME/lora-finetuning:latest
docker run --rm YOUR_USERNAME/lora-finetuning:latest
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Test (5-10 minutes) ⭐ RECOMMENDED FIRST

```bash
python unit_test.py
```

This verifies your setup works correctly before running full training.

### Step 3: Run Full Training (30-60 min GPU, 3-6 hours CPU)

```bash
python lora_finetuning.py
```

## 🧪 Unit Test Script

**File**: `unit_test.py`

**Purpose**: Standalone test to verify your environment is properly configured.

**How to Run**:
```bash
python unit_test.py
```

**What it does**:
- Loads 16 samples (10 train, 3 validation, 3 test) from distinct dataset ranges
- Applies LoRA configuration
- Trains for 1 epoch
- Evaluates before and after training
- Completes in **5-10 minutes** (CPU) or **2-3 minutes** (GPU)

**Expected Output**:
```
================================================================================
UNIT TEST RESULTS
================================================================================
Original Model Accuracy:   30-40%
Fine-tuned Model Accuracy: 50-70%
Improvement:               +20-30%
================================================================================
✓ Unit test completed successfully!
✓ Results saved to unit_test_results.json
```

**Note**: The unit test uses samples 0-15, which are **completely separate** from the full training samples (1000-1199 for validation/test).

## 📊 Sample Output Comparison

### Important Note on Data Separation

**Training Data**: Samples 0-999 (1,000 samples)  
**Validation Data**: Samples 1,000-1,099 (100 samples)  
**Test Data**: Samples 1,100-1,199 (100 samples)

✅ **Test samples were NEVER used during fine-tuning** - they are held-out for unbiased evaluation.

### Model Comparison: Free Throw Percentage Problem

#### Question

Anthony made $5$ of his first $12$ free throw attempts. If he makes $2/3$ of his next $24$ attempts, by how many percentage points will he increase his overall success rate percentage? Express your answer to the nearest whole number.

**Ground Truth Answer:** `17`

---

#### Original Model Performance

**Prediction:** `0` ❌  
**Correct:** `false`

##### Original Model's Reasoning (Incorrect)

The original model made a critical error in step 3:

1. Correctly identified initial success rate: 5/12 ≈ 41.67%
2. Correctly calculated new successful attempts: 2/3 × 24 = 16
3. **ERROR:** Calculated new success rate as 16/24 ≈ 66.67% (only considering the new attempts, not the overall rate)
4. Calculated increase: 66.67% - 41.67% = 25%
5. Final answer: ~25% (reported as "0" in prediction)

**Key Issue:** The model failed to combine the initial and new attempts to calculate the overall success rate. It compared the initial rate (5/12) with only the new rate (16/24) instead of the combined overall rate (21/36).

---

#### Finetuned Model Performance

**Prediction:** `17` ✅  
**Correct:** `true`

##### Finetuned Model's Reasoning (Correct)

The finetuned model correctly followed the complete calculation:

1. Initial success rate: 5/12 ≈ 41.67%
2. New successful attempts: 2/3 × 24 = 16
3. **CORRECT:** Combined total successful attempts: 5 + 16 = 21
4. **CORRECT:** Combined total attempts: 12 + 24 = 36
5. **CORRECT:** Overall success rate: 21/36 ≈ 58.33%
6. **CORRECT:** Percentage point increase: 58.33% - 41.67% = 16.66%
7. Rounded to nearest whole number: **17**

---

#### Key Improvement

The finetuned model demonstrates improved reasoning by:

- **Correctly aggregating** all attempts (initial + new) before calculating the overall success rate
- **Understanding the problem context** - recognizing that "overall success rate" means combining all attempts, not just comparing initial vs. new rates
- **Following the complete calculation chain** without skipping steps

This improvement shows that LoRA finetuning helped the model better understand multi-step word problems requiring cumulative calculations.


## 💡 Key Observations: Base vs Fine-tuned Model

### 1. **Answer Format Adherence**

**Base Model**: Often includes correct reasoning but fails to provide answer in the required format, or extracts wrong numbers from its own explanation.

**Fine-tuned Model**: Consistently follows "The answer is: X" format and clearly separates the final answer from intermediate calculations.

**Impact**: The fine-tuned model's structured output makes answer extraction much more reliable.

### 2. **Step-by-Step Clarity**

**Base Model**: Sometimes verbose or circular, occasionally second-guesses itself mid-solution.

**Fine-tuned Model**: Clean, numbered steps with clear progression from problem to solution.

**Impact**: Better interpretability and fewer errors in multi-step problems.

### 3. **Final Answer Selection**

**Base Model**: Sometimes provides the correct calculation but then states a different number as the final answer, or gets confused about which value to report.

**Fine-tuned Model**: Correctly identifies and reports the final answer, even in problems with multiple intermediate values.

**Impact**: 11% absolute accuracy improvement (22.9% relative improvement).

### 4. **Numerical Accuracy**

**Base Model**: 48% accuracy - nearly half of answers are incorrect due to:
- Wrong final number selected
- Calculation errors
- Format confusion

**Fine-tuned Model**: 59% accuracy - majority of answers correct due to:
- Better format following
- More reliable calculations
- Clearer final answer identification

## 📁 Output Files

After running `lora_finetuning.py`, you'll get:

1. **`lora_finetuned_model/`** - Fine-tuned model with LoRA adapters (~18MB)
2. **`original_model_results.json`** - Baseline results with full outputs
3. **`finetuned_model_results.json`** - Fine-tuned results with full outputs
4. **`comparison_results.json`** - Summary metrics

Each result file contains detailed information:
```json
{
  "query": "...",
  "ground_truth": "18",
  "ground_truth_output": "... full ground truth text ...",
  "prediction": "18",
  "model_output": "... full model generation ...",
  "correct": true
}
```

This allows you to:
- ✅ See exact model outputs (not just parsed answers)
- ✅ Verify answer extraction logic
- ✅ Analyze error patterns
- ✅ Compare reasoning quality

## 🎯 LoRA Configuration

```python
LoraConfig(
    r=16,                    # Rank (controls capacity)
    lora_alpha=32,           # Scaling (typically 2× rank)
    target_modules=[         # Apply to attention layers
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,       # Regularization
    task_type=TaskType.CAUSAL_LM
)
```

**Result**: Only 4.6M parameters trained (0.76% of 600M total) - highly efficient!

## 📖 Usage Guide

### All Available Scripts

| Script | Purpose | Runtime | Command |
|--------|---------|---------|---------|
| **`unit_test.py`** | Quick setup verification | 5-10 min | `python unit_test.py` |
| **`lora_finetuning.py`** | Full training pipeline | 30-60 min (GPU) | `python lora_finetuning.py` |
| **`evaluate_original_model.py`** | Baseline only | 10-20 min | `python evaluate_original_model.py` |
| **`inference_example.py`** | Test fine-tuned model | 1-2 min | `python inference_example.py` |

### Recommended Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit test (verify setup)
python unit_test.py

# 3. If unit test passes, run full training
python lora_finetuning.py

# 4. Test your fine-tuned model
python inference_example.py
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `lora_finetuning.py`:
```python
per_device_train_batch_size=2  # Instead of 4
```

### Slow on CPU

Expected behavior - use GPU or reduce samples:
```python
NUM_TRAIN_SAMPLES = 500  # Instead of 1000
```

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

## 📚 Technical Details

### Dataset Split Strategy

```
MetaMathQA samples 0-1199:
├── Training:    0-999    (1,000 samples) → Used for model training
├── Validation:  1000-1099 (100 samples) → Used during training for checkpointing
└── Test:        1100-1199 (100 samples) → Held-out for final evaluation ⭐
```

**Critical**: Test samples are NEVER seen during training, ensuring unbiased evaluation.

### Why LoRA?

Traditional fine-tuning requires training all 600M parameters. LoRA trains only 0.76% (4.6M parameters) by adding small matrices to attention layers, achieving:

- ✅ 80% less memory usage
- ✅ 99% less storage (18MB vs 2.4GB)
- ✅ Similar or better accuracy
- ✅ Faster training

### What Gets Trained

LoRA is applied only to attention projection layers:
- Query projection (`q_proj`)
- Key projection (`k_proj`)
- Value projection (`v_proj`)
- Output projection (`o_proj`)

All other 99.24% of parameters remain frozen.

## 📦 Installation

### Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- GPU with 6GB+ VRAM (optional but recommended)

### Dependencies

```bash
pip install -r requirements.txt
```

Core packages:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `peft>=0.7.0`
- `datasets>=2.14.0`

## 🎓 Project Structure

```
LoRA-finetuning/
├── unit_test.py              # Quick verification (5-10 min) ⭐
├── lora_finetuning.py        # Full training pipeline
├── evaluate_original_model.py # Baseline evaluation
├── inference_example.py      # Inference demo
├── Dockerfile                # Docker configuration
├── build_docker.sh           # Docker build helper
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔗 Resources

- **Repository**: https://github.com/qixuanwang-nu/LoRA-finetuning
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **PEFT Docs**: https://huggingface.co/docs/peft
- **Qwen Models**: https://huggingface.co/Qwen
- **MetaMathQA**: https://huggingface.co/datasets/meta-math/MetaMathQA
