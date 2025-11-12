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
    PeftModel
)
from datasets import load_dataset
import re
import os
from tqdm import tqdm
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and dataset configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"  # Qwen3 0.6B base model for fine-tuning
DATASET_NAME = "meta-math/MetaMathQA"
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 100
NUM_EVAL_SAMPLES = 100
OUTPUT_DIR = "./lora_finetuned_model"
ORIGINAL_EVAL_RESULTS = "./original_model_results.json"
FINETUNED_EVAL_RESULTS = "./finetuned_model_results.json"

def extract_final_answer(text):
    """
    Extract the final answer from the text.
    Looking for pattern: "The answer is: {final_answer}"
    """
    # Helper to clean a matched numeric string
    def _clean_number(num_str: str) -> str:
        s = num_str.strip()
        if s.endswith('.'):
            s = s[:-1]
        s = s.replace(',', '')
        return s

    # Regex to match numbers (including those with commas)
    number_regex = re.compile(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\.?')

    def _first_math_token(s: str):
        # Simply extract the first number found
        m = number_regex.search(s)
        if m:
            return _clean_number(m.group(0))
        return None

    # 1) Prefer the last sentence; take the FIRST math token after the word "answer" or "is"
    text_stripped = text.strip()
    if text_stripped:
        sentences = re.split(r'(?<=[.!?])\s+', text_stripped)
        last_sentence = sentences[-1] if sentences else text_stripped
        if not last_sentence.strip():
            lines = [ln for ln in text_stripped.splitlines() if ln.strip()]
            last_sentence = lines[-1] if lines else text_stripped
        # First try "answer"
        m_ans = re.search(r'answer\b(.*)$', last_sentence, re.IGNORECASE)
        if m_ans:
            after_answer = m_ans.group(1)
            token = _first_math_token(after_answer)
            if token is not None:
                return token
        # Then try "is" pattern (e.g., "the age is 38 years")
        m_is = re.search(r'\bis\b\s+(.*)$', last_sentence, re.IGNORECASE)
        if m_is:
            after_is = m_is.group(1)
            token = _first_math_token(after_is)
            if token is not None:
                return token

    # 2) Try to find "The answer is:" pattern anywhere; parse first math token in the capture
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

    # 3) Fallback: first available math token from the whole text; else last number as before
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
    # Remove commas from numbers
    answer = answer.replace(',', '')
    # Remove dollar signs and percent signs
    answer = answer.replace('$', '').replace('%', '')
    return answer

def load_and_prepare_dataset():
    """Load and prepare the MetaMathQA dataset"""
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Convert to list and take first samples
    samples = []
    for i, example in enumerate(dataset):
        if i >= NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES + NUM_EVAL_SAMPLES:
            break
        samples.append(example)

    print(f"Loaded {len(samples)} samples")

    # Split into train / validation / evaluation(test)
    train_start = 0
    train_end = NUM_TRAIN_SAMPLES
    val_end = train_end + NUM_VAL_SAMPLES
    eval_end = val_end + NUM_EVAL_SAMPLES

    train_samples = samples[train_start:train_end]
    val_samples = samples[train_end:val_end]
    eval_samples = samples[val_end:eval_end]

    return train_samples, val_samples, eval_samples

def format_instruction(query):
    """Format the instruction with explicit answer format requirement"""
    return f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""

def preprocess_function(examples, tokenizer, max_length=1024):
    """Preprocess the dataset for training"""
    # Format: instruction + response
    texts = []
    for query, response in zip(examples['query'], examples['response']):
        instruction = format_instruction(query)
        full_text = instruction + " " + response
        texts.append(full_text)

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

def evaluate_model(model, tokenizer, eval_samples, model_name="model"):
    """Evaluate model on evaluation samples"""
    print(f"\nEvaluating {model_name}...")
    model.eval()

    correct = 0
    total = len(eval_samples)
    results = []

    for sample in tqdm(eval_samples, desc=f"Evaluating {model_name}"):
        query = sample['query']
        ground_truth_response = sample['response']

        # Extract ground truth answer
        ground_truth_answer = extract_final_answer(ground_truth_response)
        ground_truth_normalized = normalize_answer(ground_truth_answer)

        # Generate prediction
        instruction = format_instruction(query)
        inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the instruction part
        prediction = prediction[len(instruction):].strip()

        # Extract predicted answer
        predicted_answer = extract_final_answer(prediction)
        predicted_normalized = normalize_answer(predicted_answer)

        # Check if correct
        is_correct = predicted_normalized == ground_truth_normalized
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "ground_truth": ground_truth_answer,
            "prediction": predicted_answer,
            "correct": is_correct
        })

    accuracy = correct / total
    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy, results

def main():
    print("="*80)
    print("LoRA Fine-tuning with PEFT for Qwen3-0.6B on MetaMathQA")
    print("="*80)

    # Load dataset
    train_samples, val_samples, eval_samples = load_and_prepare_dataset()
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    # Display a sample
    print("\n" + "="*80)
    print("Sample from dataset:")
    print(f"Query: {train_samples[0]['query'][:200]}...")
    print(f"Response: {train_samples[0]['response'][:200]}...")
    print("="*80 + "\n")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        original_model = original_model.to(device)

    print(f"Model loaded. Parameters: {original_model.num_parameters():,}")

    # Evaluate original model
    print("\n" + "="*80)
    print("EVALUATING ORIGINAL MODEL")
    print("="*80)
    original_accuracy, original_results = evaluate_model(
        original_model, tokenizer, eval_samples, "Original Model"
    )

    # Save original model results
    with open(ORIGINAL_EVAL_RESULTS, 'w') as f:
        json.dump({
            "accuracy": original_accuracy,
            "results": original_results
        }, f, indent=2)
    print(f"Original model results saved to {ORIGINAL_EVAL_RESULTS}")

    # Prepare dataset for training
    print("\n" + "="*80)
    print("PREPARING DATASET FOR TRAINING")
    print("="*80)

    # Create dataset dictionary
    train_dataset_dict = {
        'query': [s['query'] for s in train_samples],
        'response': [s['response'] for s in train_samples]
    }

    # Tokenize dataset
    print("Tokenizing training data...")
    train_encodings = preprocess_function(train_dataset_dict, tokenizer)

    # Create PyTorch dataset
    class MathDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    train_dataset = MathDataset(train_encodings)
    print(f"Training dataset prepared with {len(train_dataset)} samples")

    # Prepare validation dataset (distinct from held-out evaluation set)
    print("Tokenizing validation data...")
    val_dataset_dict = {
        'query': [s['query'] for s in val_samples],
        'response': [s['response'] for s in val_samples]
    }
    val_encodings = preprocess_function(val_dataset_dict, tokenizer)
    val_dataset = MathDataset(val_encodings)
    print(f"Validation dataset prepared with {len(val_dataset)} samples")

    # Load model for fine-tuning
    print("\n" + "="*80)
    print("SETTING UP LORA CONFIGURATION")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank adaptation
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
        lora_dropout=0.05,  # Dropout probability
        bias="none",  # Bias training strategy
        task_type=TaskType.CAUSAL_LM  # Task type
    )

    print("LoRA Configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    print(f"  Dropout: {lora_config.lora_dropout}")

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")

    # Setup training arguments
    print("\n" + "="*80)
    print("SETTING UP TRAINING")
    print("="*80)

    # Create TrainingArguments with backward compatibility across transformers versions
    try:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            warmup_steps=50,
            weight_decay=0.01,
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    except TypeError:
        # Fallback for older versions where some arguments are unsupported
        try:
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                warmup_steps=50,
                weight_decay=0.01,
            )
            print("Note: Using fallback TrainingArguments (older transformers version detected).")
        except TypeError:
            # Minimal fallback - guarantees instantiation
            training_args = TrainingArguments(output_dir=OUTPUT_DIR)
            print("Note: Using minimal TrainingArguments due to very old transformers version.")

    print("Training Arguments:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train the model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    trainer.train()

    # Save the fine-tuned model
    print("\nSaving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Evaluate fine-tuned model
    print("\n" + "="*80)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*80)

    finetuned_accuracy, finetuned_results = evaluate_model(
        model, tokenizer, eval_samples, "Fine-tuned Model"
    )

    # Save fine-tuned model results
    with open(FINETUNED_EVAL_RESULTS, 'w') as f:
        json.dump({
            "accuracy": finetuned_accuracy,
            "results": finetuned_results
        }, f, indent=2)
    print(f"Fine-tuned model results saved to {FINETUNED_EVAL_RESULTS}")

    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Original Model Accuracy: {original_accuracy:.4f}")
    print(f"Fine-tuned Model Accuracy: {finetuned_accuracy:.4f}")
    print(f"Improvement: {(finetuned_accuracy - original_accuracy):.4f} ({100 * (finetuned_accuracy - original_accuracy):.2f}%)")
    print("="*80)

    # Save comparison
    with open("comparison_results.json", 'w') as f:
        json.dump({
            "original_accuracy": original_accuracy,
            "finetuned_accuracy": finetuned_accuracy,
            "improvement": finetuned_accuracy - original_accuracy,
            "improvement_percentage": 100 * (finetuned_accuracy - original_accuracy)
        }, f, indent=2)
    print("\nComparison results saved to comparison_results.json")

    print("\n✓ Training and evaluation complete!")

if __name__ == "__main__":
    main()
