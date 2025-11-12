"""
Analyze the actual token lengths in the MetaMathQA dataset
to determine optimal max_length and max_new_tokens values
"""

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "meta-math/MetaMathQA"
NUM_SAMPLES_TO_ANALYZE = 1200  # Same as training + eval

def analyze_lengths():
    print("="*80)
    print("Analyzing Token Lengths in MetaMathQA Dataset")
    print("="*80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load dataset
    print(f"Loading {NUM_SAMPLES_TO_ANALYZE} samples from dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    samples = []
    for i, example in enumerate(dataset):
        if i >= NUM_SAMPLES_TO_ANALYZE:
            break
        samples.append(example)

    print(f"Loaded {len(samples)} samples\n")

    # Analyze lengths
    query_lengths = []
    response_lengths = []
    combined_lengths = []

    print("Tokenizing and measuring lengths...")
    for sample in tqdm(samples):
        query = sample['query']
        response = sample['response']

        # Format as it will be used in training
        instruction = f"""Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""

        full_text = instruction + " " + response

        # Tokenize
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        combined_tokens = tokenizer.encode(full_text, add_special_tokens=True)

        query_lengths.append(len(query_tokens))
        response_lengths.append(len(response_tokens))
        combined_lengths.append(len(combined_tokens))

    # Convert to numpy arrays
    query_lengths = np.array(query_lengths)
    response_lengths = np.array(response_lengths)
    combined_lengths = np.array(combined_lengths)

    # Print statistics
    print("\n" + "="*80)
    print("LENGTH STATISTICS")
    print("="*80)

    print("\n1. QUERY (Question) Lengths:")
    print(f"   Min:        {query_lengths.min():6d} tokens")
    print(f"   Max:        {query_lengths.max():6d} tokens")
    print(f"   Mean:       {query_lengths.mean():6.1f} tokens")
    print(f"   Median:     {np.median(query_lengths):6.1f} tokens")
    print(f"   75th %ile:  {np.percentile(query_lengths, 75):6.1f} tokens")
    print(f"   90th %ile:  {np.percentile(query_lengths, 90):6.1f} tokens")
    print(f"   95th %ile:  {np.percentile(query_lengths, 95):6.1f} tokens")
    print(f"   99th %ile:  {np.percentile(query_lengths, 99):6.1f} tokens")

    print("\n2. RESPONSE (Answer) Lengths:")
    print(f"   Min:        {response_lengths.min():6d} tokens")
    print(f"   Max:        {response_lengths.max():6d} tokens")
    print(f"   Mean:       {response_lengths.mean():6.1f} tokens")
    print(f"   Median:     {np.median(response_lengths):6.1f} tokens")
    print(f"   75th %ile:  {np.percentile(response_lengths, 75):6.1f} tokens")
    print(f"   90th %ile:  {np.percentile(response_lengths, 90):6.1f} tokens")
    print(f"   95th %ile:  {np.percentile(response_lengths, 95):6.1f} tokens")
    print(f"   99th %ile:  {np.percentile(response_lengths, 99):6.1f} tokens")

    print("\n3. COMBINED (Instruction + Question + Answer) Lengths:")
    print(f"   Min:        {combined_lengths.min():6d} tokens")
    print(f"   Max:        {combined_lengths.max():6d} tokens")
    print(f"   Mean:       {combined_lengths.mean():6.1f} tokens")
    print(f"   Median:     {np.median(combined_lengths):6.1f} tokens")
    print(f"   75th %ile:  {np.percentile(combined_lengths, 75):6.1f} tokens")
    print(f"   90th %ile:  {np.percentile(combined_lengths, 90):6.1f} tokens")
    print(f"   95th %ile:  {np.percentile(combined_lengths, 95):6.1f} tokens")
    print(f"   99th %ile:  {np.percentile(combined_lengths, 99):6.1f} tokens")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Recommend based on 95th percentile with some buffer
    recommended_max_length = int(np.ceil(np.percentile(combined_lengths, 95) / 128) * 128)
    recommended_max_new_tokens = int(np.ceil(np.percentile(response_lengths, 95) / 128) * 128)

    print(f"\n1. For max_length (training tokenization):")
    print(f"   95th percentile: {np.percentile(combined_lengths, 95):.0f} tokens")
    print(f"   Recommended:     {recommended_max_length} tokens (covers 95% of data)")
    print(f"   Current setting: 2048 tokens")

    coverage_at_recommended = (combined_lengths <= recommended_max_length).sum() / len(combined_lengths) * 100
    print(f"   Coverage:        {coverage_at_recommended:.1f}% of samples")

    print(f"\n2. For max_new_tokens (generation):")
    print(f"   95th percentile: {np.percentile(response_lengths, 95):.0f} tokens")
    print(f"   Recommended:     {recommended_max_new_tokens} tokens (covers 95% of responses)")
    print(f"   Current setting: 2048 tokens")

    coverage_at_recommended_gen = (response_lengths <= recommended_max_new_tokens).sum() / len(response_lengths) * 100
    print(f"   Coverage:        {coverage_at_recommended_gen:.1f}% of responses")

    # Memory impact
    print("\n3. Memory Impact:")
    memory_ratio = 2048 / recommended_max_length
    print(f"   Current (2048) vs Recommended ({recommended_max_length}):")
    print(f"   Memory usage:    {memory_ratio:.1f}x higher with current setting")
    print(f"   Training speed:  ~{memory_ratio:.1f}x slower with current setting")

    # Distribution analysis
    print("\n4. Length Distribution:")
    ranges = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, float('inf'))]
    for low, high in ranges:
        count = ((combined_lengths >= low) & (combined_lengths < high)).sum()
        pct = count / len(combined_lengths) * 100
        if high == float('inf'):
            print(f"   {low:4d}+ tokens:      {count:4d} samples ({pct:5.1f}%)")
        else:
            print(f"   {low:4d}-{high:4d} tokens: {count:4d} samples ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("CONFIGURATION SUGGESTION")
    print("="*80)
    print(f"\nUpdate your config.py with:")
    print(f"   MAX_SEQUENCE_LENGTH = {recommended_max_length}  # Current: 2048")
    print(f"   max_new_tokens = {recommended_max_new_tokens}      # Current: 2048")
    print(f"\nThis will:")
    print(f"   - Cover ~95% of your training data")
    print(f"   - Reduce memory usage by ~{memory_ratio:.1f}x")
    print(f"   - Speed up training by ~{memory_ratio:.1f}x")
    print(f"   - Still handle most long responses")

    # Show some examples of long samples
    print("\n" + "="*80)
    print("EXAMPLES OF LONGEST SAMPLES")
    print("="*80)

    # Get indices of top 3 longest
    longest_indices = np.argsort(combined_lengths)[-3:]

    for i, idx in enumerate(longest_indices[::-1], 1):
        print(f"\n{i}. Sample with {combined_lengths[idx]} tokens:")
        print(f"   Query: {samples[idx]['query'][:150]}...")
        print(f"   Response length: {response_lengths[idx]} tokens")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

    return {
        'query_lengths': query_lengths,
        'response_lengths': response_lengths,
        'combined_lengths': combined_lengths,
        'recommended_max_length': recommended_max_length,
        'recommended_max_new_tokens': recommended_max_new_tokens
    }

if __name__ == "__main__":
    analyze_lengths()
