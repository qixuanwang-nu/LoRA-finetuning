"""
Standalone script to evaluate the original Qwen3-0.6B model
on the MetaMathQA evaluation dataset without training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "meta-math/MetaMathQA"
NUM_TRAIN_SAMPLES = 500  # Skip these for evaluation
NUM_EVAL_SAMPLES = 50    # Evaluate on these
OUTPUT_FILE = "./original_model_evaluation.json"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_final_answer(text):
    """
    Extract the final answer from the text.
    Looking for pattern: "The answer is: {final_answer}"
    """
    # Helpers to clean tokens and normalize pi
    def _clean_number(num_str: str) -> str:
        s = num_str.strip()
        if s.endswith('.'):
            s = s[:-1]
        s = s.replace(',', '')
        return s

    def _normalize_pi_token(token: str) -> str:
        t = token.strip()
        if t.endswith('.'):
            t = t[:-1]
        t = t.replace(',', '')
        t = re.sub(r'\s*\\pi', r'\\pi', t)
        return t

    # Regexes
    number_regex = re.compile(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\.?')
    latex_frac_regex = re.compile(r'\\frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}')
    # Match slash fractions including those with parentheses, e.g., 81/(2\pi) or 3/2
    slash_frac_regex = re.compile(
        r'([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*\\pi)?|\\pi))\s*/\s*'
        r'\(?\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*\\pi)?|\\pi))\s*\)?'
    )
    pi_token_regex = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\\pi|\\pi)')

    def _first_math_token(s: str):
        m = latex_frac_regex.search(s)
        if m:
            num = _normalize_pi_token(m.group(1))
            den = _normalize_pi_token(m.group(2))
            return f"\\frac{{{num}}}{{{den}}}"
        m = slash_frac_regex.search(s)
        if m:
            num = _normalize_pi_token(m.group(1))
            den = _normalize_pi_token(m.group(2))
            return f"\\frac{{{num}}}{{{den}}}"
        m = pi_token_regex.search(s)
        if m:
            return _normalize_pi_token(m.group(0))
        m = number_regex.search(s)
        if m:
            return _clean_number(m.group(0))
        return None

    # Prefer last sentence and first math token after "answer" or "is"
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

    # Try common patterns, parse first math token within capture
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

    # Fallback: first math token in whole text, else last number
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


def format_instruction(query):
    """Format the instruction with explicit answer format requirement"""
    return f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""


def load_evaluation_dataset():
    """Load the evaluation portion of the MetaMathQA dataset"""
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Convert to list and take samples for evaluation
    samples = []
    for i, example in enumerate(dataset):
        if i >= NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES:
            break
        samples.append(example)

    print(f"Loaded {len(samples)} samples total")

    # Take evaluation samples (skip training samples)
    eval_samples = samples[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES]
    print(f"Using {len(eval_samples)} samples for evaluation")

    return eval_samples


def evaluate_model(model, tokenizer, eval_samples):
    """Evaluate model on evaluation samples"""
    print("\nEvaluating Original Qwen3-0.6B Model...")
    print("="*80)

    model.eval()

    correct = 0
    total = len(eval_samples)
    results = []

    for idx, sample in enumerate(tqdm(eval_samples, desc="Evaluating"), 1):
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

        # Store result
        result = {
            "sample_id": idx,
            "query": query,
            "ground_truth_response": ground_truth_response,
            "ground_truth_answer": ground_truth_answer,
            "model_prediction": prediction,
            "predicted_answer": predicted_answer,
            "correct": is_correct
        }
        results.append(result)

        # Print sample results every 20 samples
        if idx % 20 == 0:
            print(f"\n[Sample {idx}]")
            print(f"Query: {query[:100]}...")
            print(f"Ground Truth: {ground_truth_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print(f"Running Accuracy: {correct}/{idx} = {100*correct/idx:.2f}%")

    accuracy = correct / total

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print("="*80)

    return accuracy, results


def main():
    print("="*80)
    print("Evaluating Original Qwen3-0.6B Model on MetaMathQA")
    print("="*80)
    print()

    # Load evaluation dataset
    eval_samples = load_evaluation_dataset()

    # Display a sample
    print("\n" + "="*80)
    print("Sample from evaluation dataset:")
    print(f"Query: {eval_samples[0]['query'][:200]}...")
    print(f"Response: {eval_samples[0]['response'][:200]}...")
    print("="*80 + "\n")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    print(f"Model loaded. Total parameters: {model.num_parameters():,}")
    print()

    # Evaluate model
    accuracy, results = evaluate_model(model, tokenizer, eval_samples)

    # Save results
    output_data = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "accuracy": accuracy,
        "correct_count": sum(1 for r in results if r['correct']),
        "total_count": len(results),
        "results": results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
