"""
Example script for using the fine-tuned LoRA model for inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

def extract_final_answer(text):
    """Extract the final answer from the text."""
    # Helper to clean a matched numeric string and normalize pi tokens
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


def load_finetuned_model(model_path="./lora_finetuned_model"):
    """Load the fine-tuned LoRA model"""
    print(f"Loading fine-tuned model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)

    print("Model loaded successfully!")
    return model, tokenizer


def solve_math_problem(model, tokenizer, problem):
    """Solve a math problem using the fine-tuned model"""
    # Format the instruction
    instruction = f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {problem}

Solution:"""

    # Tokenize
    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
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

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the instruction part
    response = response[len(instruction):].strip()

    return response


def main():
    print("="*80)
    print("Fine-tuned LoRA Model Inference Example")
    print("="*80)
    print()

    # Load model
    model, tokenizer = load_finetuned_model()

    # Example problems
    test_problems = [
        "What is 15 + 27?",
        "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
        "Calculate 12 × 8",
        "A rectangle has a length of 10 cm and a width of 5 cm. What is its area?",
    ]

    print("\nTesting fine-tuned model on example problems:")
    print("="*80)

    for i, problem in enumerate(test_problems, 1):
        print(f"\nProblem {i}: {problem}")
        print("-"*80)

        # Solve problem
        solution = solve_math_problem(model, tokenizer, problem)
        print(f"Solution:\n{solution}")

        # Extract answer
        answer = extract_final_answer(solution)
        print(f"\nExtracted Answer: {answer}")
        print("="*80)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
