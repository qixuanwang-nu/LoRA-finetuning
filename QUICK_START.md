# Quick Start Guide

## To Evaluate the Original Model Only

### Why do this?
- Get baseline accuracy in ~10-20 minutes
- Verify setup before long training
- See what improvement to expect

### How to run:

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run evaluation
./run_original_eval.sh
```

Or:

```bash
python evaluate_original_model.py
```

### What you'll get:
- **File**: `original_model_evaluation.json`
- **Content**:
  - Accuracy on 200 test questions
  - Detailed results for each question
  - Ground truth vs predicted answers

### Example output:
```
Evaluating Original Qwen3-0.6B Model
======================================
Using 200 samples for evaluation
Evaluating: 100%|████████| 200/200

EVALUATION COMPLETE
======================================
Total Samples: 200
Correct: 45
Incorrect: 155
Accuracy: 0.2250 (22.50%)
======================================

Results saved to: original_model_evaluation.json
```

---

## To Run Complete Training Pipeline

### What this does:
1. Evaluates original model (baseline)
2. LoRA fine-tunes on 1,000 samples (~1-3 hours)
3. Evaluates fine-tuned model
4. Compares performance

### How to run:

```bash
./run_training.sh
```

Or:

```bash
python lora_finetuning.py
```

### What you'll get:
- `lora_finetuned_model/` - Fine-tuned model
- `original_model_results.json` - Baseline results
- `finetuned_model_results.json` - Fine-tuned results
- `comparison_results.json` - Accuracy improvement

---

## Understanding the Results

### Evaluation JSON Structure:

```json
{
  "model_name": "Qwen/Qwen3-0.6B-Base",
  "accuracy": 0.2250,
  "correct_count": 45,
  "total_count": 200,
  "results": [
    {
      "sample_id": 1,
      "query": "What is 15 + 27?",
      "ground_truth_answer": "42",
      "predicted_answer": "42",
      "correct": true
    },
    ...
  ]
}
```

### Key Metrics:

- **accuracy**: Percentage of correct answers (0.0 to 1.0)
- **correct_count**: Number of correct predictions
- **total_count**: Total questions evaluated (200)

---

## Modifying Evaluation Size

To evaluate on more/fewer questions, edit:

**In `evaluate_original_model.py`:**
```python
NUM_EVAL_SAMPLES = 200  # Change this number
```

**In `lora_finetuning.py`:**
```python
NUM_TRAIN_SAMPLES = 1000  # Training samples
NUM_EVAL_SAMPLES = 100    # Evaluation samples
```

Or use `config.py` for centralized configuration.

---

## Expected Accuracy

### Original Qwen3-0.6B-Base:
- **Expected**: 15-30% on MetaMathQA
- Small base model, not trained specifically for math

### After LoRA Fine-tuning:
- **Expected**: 40-70% on MetaMathQA
- **Improvement**: +25-40 percentage points
- Better format compliance ("The answer is: X")

---

## Troubleshooting

### "Out of memory" error:
```python
# Reduce batch size in lora_finetuning.py
per_device_train_batch_size=2  # was 4
```

### Evaluation taking too long:
```python
# Reduce evaluation samples
NUM_EVAL_SAMPLES = 50  # was 200
```

### Import errors:
```bash
pip install --upgrade -r requirements.txt
```

---

## Next Steps After Evaluation

1. **View detailed results**:
   ```bash
   cat original_model_evaluation.json | python -m json.tool | less
   ```

2. **Check specific errors**:
   ```python
   import json
   with open('original_model_evaluation.json') as f:
       data = json.load(f)

   # See incorrect predictions
   errors = [r for r in data['results'] if not r['correct']]
   print(f"Found {len(errors)} errors")
   ```

3. **Run training** if satisfied with setup:
   ```bash
   ./run_training.sh
   ```

4. **Test fine-tuned model**:
   ```bash
   python inference_example.py
   ```
