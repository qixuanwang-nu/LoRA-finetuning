"""
Unit Test Script for LoRA Fine-tuning
Tests the fine-tuning pipeline on a minimal dataset to verify setup quickly.
Expected runtime: 5-10 minutes (CPU) or 2-3 minutes (GPU)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import load_dataset
import re
import json
from datetime import datetime

# Test configuration - minimal for quick testing
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "meta-math/MetaMathQA"
NUM_TRAIN_SAMPLES = 10  # Very small for quick testing
NUM_VAL_SAMPLES = 3
NUM_EVAL_SAMPLES = 3
OUTPUT_DIR = "./unit_test_model"

print("=" * 80)
print("LoRA Fine-tuning Unit Test")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nThis is a quick test with minimal data to verify your setup works.")
print(f"Training samples: {NUM_TRAIN_SAMPLES}")
print(f"Validation samples: {NUM_VAL_SAMPLES}")
print(f"Evaluation samples: {NUM_EVAL_SAMPLES}")
print(f"Expected runtime: 5-10 minutes (CPU) or 2-3 minutes (GPU)")
print("=" * 80)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

def extract_final_answer(text):
    """Extract the final answer from the text."""
    def _clean_number(num_str: str) -> str:
        s = num_str.strip()
        if s.endswith('.'):
            s = s[:-1]
        s = s.replace(',', '')
        return s

    number_regex = re.compile(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\.?')

    def _first_math_token(s: str):
        m = number_regex.search(s)
        if m:
            return _clean_number(m.group(0))
        return None

    text_stripped = text.strip()
    if text_stripped:
        sentences = re.split(r'(?<=[.!?])\s+', text_stripped)
        last_sentence = sentences[-1] if sentences else text_stripped
        if not last_sentence.strip():
            lines = [ln for ln in text_stripped.splitlines() if ln.strip()]
            last_sentence = lines[-1] if lines else text_stripped
        
        m_ans = re.search(r'\banswer\b[:\s]*(.*)$', last_sentence, re.IGNORECASE)
        if m_ans:
            after_answer = m_ans.group(1)
            token = _first_math_token(after_answer)
            if token is not None:
                return token
        
        m_is = re.search(r'\bis\b\s+(.*)$', last_sentence, re.IGNORECASE)
        if m_is:
            after_is = m_is.group(1)
            token = _first_math_token(after_is)
            if token is not None:
                return token

    patterns = [
        r"The answer is:\s*([^\n.]+)",
        r"The answer is\s*:\s*([^\n.]+)",
        r"####\s*([^\n]+)",
        r"the answer is:\s*([^\n.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            captured = match.group(1).strip()
            token = _first_math_token(captured)
            if token is not None:
                return token
            return captured.rstrip('.')

    token = _first_math_token(text)
    if token is not None:
        return token

    all_nums = number_regex.findall(text)
    if all_nums:
        return _clean_number(all_nums[-1])

    return text_stripped

def normalize_answer(answer):
    """Normalize answer for comparison"""
    answer = str(answer).strip().lower()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '').replace('%', '')
    return answer

def format_instruction(query):
    """Format the instruction"""
    return f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""

def load_test_dataset():
    """Load minimal dataset for testing"""
    print("\n[1/6] Loading test dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    samples = []
    total_needed = NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES + NUM_EVAL_SAMPLES
    for i, example in enumerate(dataset):
        if i >= total_needed:
            break
        samples.append(example)
    
    train_samples = samples[:NUM_TRAIN_SAMPLES]
    val_samples = samples[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES]
    eval_samples = samples[NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES:total_needed]
    
    print(f"   ✓ Loaded {len(samples)} samples")
    print(f"   - Train: {len(train_samples)}")
    print(f"   - Validation: {len(val_samples)}")
    print(f"   - Test: {len(eval_samples)}")
    
    return train_samples, val_samples, eval_samples

def evaluate_sample(model, tokenizer, sample):
    """Evaluate a single sample"""
    query = sample['query']
    ground_truth_response = sample['response']
    
    ground_truth_answer = extract_final_answer(ground_truth_response)
    ground_truth_normalized = normalize_answer(ground_truth_answer)
    
    instruction = format_instruction(query)
    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = prediction[len(instruction):].strip()
    
    predicted_answer = extract_final_answer(prediction)
    predicted_normalized = normalize_answer(predicted_answer)
    
    is_correct = predicted_normalized == ground_truth_normalized
    return is_correct, predicted_answer, ground_truth_answer

def quick_evaluate(model, tokenizer, eval_samples, model_name="model"):
    """Quick evaluation on small dataset"""
    print(f"   Evaluating {model_name} on {len(eval_samples)} samples...")
    model.eval()
    
    correct = 0
    for sample in eval_samples:
        is_correct, _, _ = evaluate_sample(model, tokenizer, sample)
        if is_correct:
            correct += 1
    
    accuracy = correct / len(eval_samples) if eval_samples else 0
    print(f"   ✓ Accuracy: {accuracy:.2%} ({correct}/{len(eval_samples)})")
    return accuracy

def main():
    try:
        # Load dataset
        train_samples, val_samples, eval_samples = load_test_dataset()
        
        # Load tokenizer
        print("\n[2/6] Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            model = model.to(device)
        
        print(f"   ✓ Model loaded ({model.num_parameters():,} parameters)")
        
        # Quick baseline evaluation
        print("\n[3/6] Baseline evaluation (original model)...")
        original_accuracy = quick_evaluate(model, tokenizer, eval_samples, "Original Model")
        
        # Prepare training data
        print("\n[4/6] Preparing training data...")
        train_texts = []
        for sample in train_samples:
            instruction = format_instruction(sample['query'])
            full_text = instruction + " " + sample['response']
            train_texts.append(full_text)
        
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        train_encodings["labels"] = train_encodings["input_ids"].clone()
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __len__(self):
                return len(self.encodings['input_ids'])
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
        
        train_dataset = SimpleDataset(train_encodings)
        print(f"   ✓ Training dataset ready ({len(train_dataset)} samples)")
        
        # Apply LoRA
        print("\n[5/6] Applying LoRA and training...")
        lora_config = LoraConfig(
            r=8,  # Smaller rank for faster testing
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ LoRA applied ({trainable_params:,} trainable params, {100*trainable_params/total_params:.2f}%)")
        
        # Minimal training
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,  # Just 1 epoch for quick test
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=5,
            save_strategy="no",  # Don't save checkpoints in test
            report_to="none",
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        print("   Training for 1 epoch (this will take a few minutes)...")
        trainer.train()
        print("   ✓ Training complete")
        
        # Quick post-training evaluation
        print("\n[6/6] Post-training evaluation...")
        finetuned_accuracy = quick_evaluate(model, tokenizer, eval_samples, "Fine-tuned Model")
        
        # Results
        print("\n" + "=" * 80)
        print("UNIT TEST RESULTS")
        print("=" * 80)
        print(f"Original Model Accuracy:   {original_accuracy:.2%}")
        print(f"Fine-tuned Model Accuracy: {finetuned_accuracy:.2%}")
        improvement = finetuned_accuracy - original_accuracy
        print(f"Improvement:               {improvement:+.2%}")
        print("=" * 80)
        
        # Save test results
        test_results = {
            "test_date": datetime.now().isoformat(),
            "num_train_samples": NUM_TRAIN_SAMPLES,
            "num_eval_samples": NUM_EVAL_SAMPLES,
            "original_accuracy": original_accuracy,
            "finetuned_accuracy": finetuned_accuracy,
            "improvement": improvement,
            "device": str(device),
            "model_name": MODEL_NAME
        }
        
        with open("unit_test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print("\n✓ Unit test completed successfully!")
        print(f"✓ Results saved to unit_test_results.json")
        print(f"✓ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nYour setup is working correctly! You can now run the full training with:")
        print("  python lora_finetuning.py")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ UNIT TEST FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (pip install -r requirements.txt)")
        print("  2. You have sufficient memory (at least 4GB RAM)")
        print("  3. Internet connection is available (for downloading model/data)")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

