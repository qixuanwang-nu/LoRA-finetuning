"""
Configuration file for LoRA fine-tuning
Modify these parameters to customize the training
"""

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "meta-math/MetaMathQA"

# =============================================================================
# Dataset Configuration
# =============================================================================
NUM_TRAIN_SAMPLES = 500  # Number of samples for training
NUM_EVAL_SAMPLES = 50    # Number of samples for evaluation
MAX_SEQUENCE_LENGTH = 1024 # Maximum sequence length for tokenization

# =============================================================================
# LoRA Configuration
# =============================================================================
LORA_CONFIG = {
    "r": 16,                    # Rank of low-rank adaptation (higher = more capacity, more parameters)
    "lora_alpha": 32,           # Scaling factor (typically 2x of r)
    "target_modules": [         # Which modules to apply LoRA to
        "q_proj",               # Query projection
        "k_proj",               # Key projection
        "v_proj",               # Value projection
        "o_proj"                # Output projection
    ],
    "lora_dropout": 0.05,       # Dropout probability for LoRA layers
    "bias": "none",             # Bias training strategy: "none", "all", or "lora_only"
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING_CONFIG = {
    "num_train_epochs": 3,                  # Number of training epochs
    "per_device_train_batch_size": 4,       # Batch size per device
    "gradient_accumulation_steps": 4,        # Gradient accumulation steps (effective batch = batch_size * grad_accum)
    "learning_rate": 2e-4,                  # Learning rate
    "warmup_steps": 50,                     # Number of warmup steps
    "weight_decay": 0.01,                   # Weight decay for regularization
    "logging_steps": 10,                    # Log every N steps
    "save_strategy": "epoch",               # Save strategy: "steps", "epoch", or "no"
    "save_total_limit": 2,                  # Maximum number of checkpoints to keep
    "fp16": True,                           # Use FP16 mixed precision (auto-detected based on CUDA)
}

# =============================================================================
# Generation Configuration (for evaluation)
# =============================================================================
GENERATION_CONFIG = {
    "max_new_tokens": 1024,     # Maximum number of tokens to generate
    "temperature": 0.7,         # Sampling temperature (lower = more deterministic)
    "top_p": 0.9,              # Nucleus sampling parameter
    "do_sample": True,         # Whether to use sampling (vs greedy decoding)
}

# =============================================================================
# Output Paths
# =============================================================================
OUTPUT_DIR = "./lora_finetuned_model"
ORIGINAL_EVAL_RESULTS = "./original_model_results.json"
FINETUNED_EVAL_RESULTS = "./finetuned_model_results.json"
COMPARISON_RESULTS = "./comparison_results.json"

# =============================================================================
# Prompt Template
# =============================================================================
PROMPT_TEMPLATE = """Solve the following math problem step by step. At the end of your solution, provide your final answer in the exact format:
The answer is: [your answer]

Problem: {query}

Solution:"""

# =============================================================================
# Answer Extraction Patterns
# =============================================================================
ANSWER_PATTERNS = [
    r"The answer is:\s*([^\n.]+)",
    r"The answer is\s*:\s*([^\n.]+)",
    r"####\s*([^\n]+)",
    r"the answer is:\s*([^\n.]+)",
]
