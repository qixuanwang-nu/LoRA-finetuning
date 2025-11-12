# LoRA Fine-tuning Implementation Notes

## Overview

This document provides detailed technical notes on the LoRA fine-tuning implementation for the Qwen3-0.6B model on the MetaMathQA dataset.

## Key Implementation Details

### 1. Dataset Handling

#### Dataset: meta-math/MetaMathQA
- **Source**: HuggingFace datasets library
- **Structure**: Each example contains:
  - `query`: The math problem/question
  - `response`: The step-by-step solution with final answer
- **Format**: Streaming dataset (large size)
- **Sampling**: First 1,000 samples for training, next 200 for evaluation

#### Data Preprocessing
```python
def format_instruction(query):
    """Formats problem with explicit answer format requirement"""
    return f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""
```

The instruction explicitly tells the model:
1. To solve step-by-step
2. To use the exact format "The answer is: [answer]"
3. Provides clear structure with Problem/Solution sections

### 2. Answer Parsing and Evaluation

#### Challenge
The model output needs to be parsed to extract the final answer and compare it with ground truth.

#### Solution: Multi-pattern Extraction
```python
def extract_final_answer(text):
    """Extract answer using multiple regex patterns"""
    patterns = [
        r"The answer is:\s*([^\n.]+)",      # Standard format
        r"The answer is\s*:\s*([^\n.]+)",   # Flexible spacing
        r"####\s*([^\n]+)",                 # Alternative format
        r"the answer is:\s*([^\n.]+)",      # Case insensitive
    ]
    # Try each pattern, fallback to extracting numbers
```

#### Normalization
```python
def normalize_answer(answer):
    """Normalize for comparison"""
    answer = str(answer).strip().lower()
    answer = answer.replace(',', '')          # Remove commas
    answer = answer.replace('$', '').replace('%', '')  # Remove symbols
    return answer
```

This handles variations like:
- "1,000" vs "1000"
- "$50" vs "50"
- "25%" vs "25"

### 3. LoRA Configuration

#### Why LoRA?
- **Efficiency**: Fine-tunes <1% of parameters
- **Performance**: Maintains or improves model quality
- **Memory**: Significantly reduces memory requirements
- **Speed**: Faster training than full fine-tuning

#### Configuration Parameters

```python
LoraConfig(
    r=16,                    # Rank: Controls capacity
    lora_alpha=32,          # Alpha: Scaling factor (typically 2x rank)
    target_modules=[        # Which attention layers to modify
        "q_proj",           # Query projections
        "k_proj",           # Key projections
        "v_proj",           # Value projections
        "o_proj"            # Output projections
    ],
    lora_dropout=0.05,      # Regularization
    bias="none",            # Don't train bias terms
    task_type=TaskType.CAUSAL_LM
)
```

#### Parameter Trade-offs

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| **r (rank)** | Fewer trainable params, faster, less capacity | More trainable params, slower, more capacity |
| **lora_alpha** | Less influence from LoRA | More influence from LoRA |
| **dropout** | Less regularization | More regularization |

**Recommended ranges**:
- r: 8-64 (we use 16)
- alpha: r to 4*r (we use 2*r = 32)
- dropout: 0.05-0.1

### 4. Training Configuration

#### Key Training Parameters

```python
TrainingArguments(
    num_train_epochs=3,                      # 3 epochs is typical for fine-tuning
    per_device_train_batch_size=4,           # Small batch for memory efficiency
    gradient_accumulation_steps=4,            # Effective batch = 4*4 = 16
    learning_rate=2e-4,                      # Higher than pre-training (1e-5)
    warmup_steps=50,                         # Gradual learning rate increase
    weight_decay=0.01,                       # L2 regularization
    fp16=True,                               # Mixed precision (if GPU available)
)
```

#### Why These Values?

1. **Batch Size (4)**:
   - Smaller batch size reduces memory usage
   - Allows running on consumer GPUs or even CPU

2. **Gradient Accumulation (4)**:
   - Simulates larger batch size (16) without memory cost
   - Stabilizes training

3. **Learning Rate (2e-4)**:
   - Higher than pre-training since we're fine-tuning
   - LoRA typically uses 1e-4 to 5e-4

4. **Epochs (3)**:
   - Prevents overfitting on small dataset
   - 1-5 epochs typical for fine-tuning

### 5. Evaluation Strategy

#### Two-Phase Evaluation

**Phase 1: Original Model Baseline**
```python
original_accuracy, original_results = evaluate_model(
    original_model, tokenizer, eval_samples, "Original Model"
)
```

**Phase 2: Fine-tuned Model**
```python
finetuned_accuracy, finetuned_results = evaluate_model(
    model, tokenizer, eval_samples, "Fine-tuned Model"
)
```

#### Evaluation Metrics

1. **Accuracy**: Percentage of correct answers
   ```python
   accuracy = correct / total
   ```

2. **Per-Sample Results**: Stored for analysis
   ```python
   {
       "query": problem,
       "ground_truth": expected_answer,
       "prediction": model_answer,
       "correct": boolean
   }
   ```

### 6. Model Architecture: Qwen3-0.6B-Base

#### Model Specifications
- **Parameters**: 0.44B non-embedding parameters
- **Layers**: 28
- **Attention Heads**: 16 (Q), 8 (KV) - uses GQA (Grouped Query Attention)
- **Context Length**: 32,768 tokens
- **Pre-training**: 36 trillion tokens across 119 languages

#### Why Base Model vs Instruct?
- **Base models** are better for fine-tuning
- **Instruct models** are already tuned for following instructions
- Fine-tuning base model gives more control over behavior

### 7. Memory Optimization Strategies

#### Implemented Optimizations

1. **LoRA**: Reduces trainable parameters by >99%
2. **FP16**: Halves memory usage with mixed precision
3. **Gradient Accumulation**: Simulates larger batch without memory cost
4. **Streaming Dataset**: Loads data incrementally
5. **Device Mapping**: Automatic GPU/CPU allocation

#### Memory Requirements (Approximate)

| Configuration | Memory Required |
|--------------|-----------------|
| Full fine-tuning (FP32) | ~16 GB |
| Full fine-tuning (FP16) | ~8 GB |
| LoRA fine-tuning (FP16) | ~4 GB |
| LoRA fine-tuning (CPU) | ~8 GB |

### 8. Expected Results

#### Performance Expectations

For mathematical reasoning tasks after LoRA fine-tuning:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Accuracy | 10-30% | 40-70% | +30-40% |
| Format compliance | Low | High | Significant |
| Step-by-step reasoning | Variable | Consistent | Improved |

**Note**: Actual results depend on:
- Dataset quality and size
- Model initialization
- Hyperparameter choices
- Task difficulty

### 9. Common Issues and Solutions

#### Issue 1: OOM (Out of Memory)
**Solutions**:
- Reduce `per_device_train_batch_size`
- Reduce `MAX_SEQUENCE_LENGTH`
- Increase `gradient_accumulation_steps`
- Use CPU (slower but works)

#### Issue 2: Poor Answer Format Compliance
**Solutions**:
- Strengthen instruction prompt
- Increase training epochs
- Add more examples with correct format
- Adjust temperature during generation

#### Issue 3: Low Accuracy Improvement
**Solutions**:
- Increase LoRA rank (r)
- Train for more epochs
- Adjust learning rate
- Use more training samples
- Try different target modules

#### Issue 4: Overfitting
**Solutions**:
- Reduce number of epochs
- Increase dropout
- Add weight decay
- Use more training data

### 10. Extension Ideas

#### Possible Improvements

1. **Advanced Answer Extraction**: Use a separate small model to extract answers
2. **Multi-task Training**: Include other math datasets
3. **Curriculum Learning**: Start with easier problems, progress to harder ones
4. **Ensemble**: Combine multiple LoRA adapters
5. **Few-shot Prompting**: Add examples in the prompt
6. **Chain-of-Thought**: Explicit reasoning step markers
7. **Reward Modeling**: Use RL to improve accuracy

#### Hyperparameter Tuning

Consider tuning:
- LoRA rank (r): [8, 16, 32, 64]
- Learning rate: [1e-4, 2e-4, 5e-4]
- Batch size: [2, 4, 8]
- Epochs: [2, 3, 5]
- Alpha to rank ratio: [1, 2, 4]

### 11. Code Structure

```
LoRA-HW2/
├── lora_finetuning.py       # Main training script
├── inference_example.py      # Inference examples
├── config.py                 # Configuration parameters
├── requirements.txt          # Python dependencies
├── run_training.sh          # Bash script to run training
├── README.md                # User documentation
└── IMPLEMENTATION_NOTES.md  # This file
```

### 12. Reproducibility

To ensure reproducible results:

1. **Set Random Seeds**:
   ```python
   torch.manual_seed(42)
   ```

2. **Fix Data Order**: Use same data samples
3. **Document Environment**: CUDA version, PyTorch version, etc.
4. **Save Hyperparameters**: Keep track of all settings

### 13. Citation and References

#### Key Papers
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Qwen**: "Qwen Technical Report" (2024)
- **PEFT**: HuggingFace PEFT library documentation

#### Useful Links
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## Conclusion

This implementation provides a complete, production-ready LoRA fine-tuning pipeline with:
- Robust answer extraction and evaluation
- Memory-efficient training
- Comprehensive logging and comparison
- Easy-to-modify configuration
- Clear documentation

The code follows best practices for fine-tuning LLMs and can be adapted for other tasks beyond mathematical reasoning.
