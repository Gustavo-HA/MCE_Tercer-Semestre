"""
Script to calculate perplexity of the fine-tuned Mistral model on the test set.
"""

import argparse
import json
import logging
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, get_chat_template
from utils.perplexity_unsloth import ppl_model, add_to_comparison, print_model_comparison
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set offline mode BEFORE importing any HuggingFace libraries
# These environment variables must be set before the libraries are imported
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def load_test_dataset(dataset_path, tokenizer):
    """Load the test dataset from JSON file and convert to Dataset format.
    
    Uses the same Alpaca chat template format as used during fine-tuning.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Format using the same chat template as fine-tuning
    texts = []
    for item in data:
        # Combine instruction and input into the user message
        user_message = item["instruction"]
        if item["input"]:
            user_message += f"\n\n{item['input']}"
        
        # Create conversation format
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": item["output"]}
        ]
        
        # Apply chat template (same as in fine-tuning)
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        texts.append(formatted_text)
    
    # Create a dataset with the text field
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity of the fine-tuned model on test set")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="data/text_gen/test_lyrics_alpaca.json",
        help="Path to the test dataset (Alpaca JSON format)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=[None, "float16", "bfloat16"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        default=False,
        help="Also evaluate the base Mistral 7B v0.3 Instruct model for comparison"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        help="Base model name to compare against (default: unsloth/mistral-7b-instruct-v0.3-bnb-4bit)"
    )
    
    args = parser.parse_args()
    
    logging.info("PERPLEXITY CALCULATION")
    logging.info("="*80)
    logging.info(f"Fine-tuned model: {args.model_dir}")
    if args.compare_base:
        logging.info(f"Base model: {args.base_model_name}")
    logging.info(f"Test dataset: {args.test_dataset}")
    logging.info(f"Max sequence length: {args.max_seq_length}")
    logging.info("="*80)
    
    # Evaluate Base Model if requested
    if args.compare_base:
        logging.info("EVALUATING BASE MODEL")
        
        # Workaround for Unsloth offline loading: Use the local cache path directly
        # Unsloth's from_pretrained uses HfFileSystem which tries to connect online even with local_files_only=True
        # Solution: Point directly to the cached snapshot directory
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = args.base_model_name.replace("/", "--")
        model_cache_path = f"{cache_dir}/models--{model_cache_name}"
        
        # Check if model exists in cache and get the snapshot path
        if os.path.exists(model_cache_path):
            refs_main = f"{model_cache_path}/refs/main"
            if os.path.exists(refs_main):
                with open(refs_main, 'r') as f:
                    commit_hash = f.read().strip()
                model_path = f"{model_cache_path}/snapshots/{commit_hash}"
                logging.info(f"Found cached model at: {model_path}")
            else:
                logging.warning(f"Cache exists but no 'main' ref found at: {refs_main}")
                model_path = args.base_model_name  # Fallback to online mode
        else:
            logging.warning(f"Model not found in cache at: {model_cache_path}")
            model_path = args.base_model_name

        
        logging.info("[1/3] Loading base model and tokenizer...")
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )
        
        # Apply the same chat template
        base_tokenizer = get_chat_template(
            base_tokenizer,
            chat_template="alpaca",
        )
        
        base_model.eval()
        logging.info("Base model loaded successfully!")
        
        logging.info("[2/3] Loading test dataset...")
        base_test_dataset = load_test_dataset(args.test_dataset, base_tokenizer)
        logging.info(f"Test dataset loaded with {len(base_test_dataset)} examples")
        
        logging.info("[3/3] Calculating perplexity for base model...")
        base_perplexity = ppl_model(base_model, base_tokenizer, base_test_dataset)
        
        # Add to comparison
        add_to_comparison("Base: Mistral-7B-Instruct-v0.3", base_perplexity)
        
        logging.info(f"Base model perplexity: {base_perplexity.item():.4f}")
        
        # Clean up base model to free memory
        del base_model, base_tokenizer, base_test_dataset
        torch.cuda.empty_cache()
    
    # Evaluate Fine-tuned Model
    logging.info("EVALUATING FINE-TUNED MODEL")
    
    logging.info("[1/3] Loading fine-tuned model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Apply the same chat template as used during fine-tuning
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="alpaca",
    )
    
    # Ensure the model is in evaluation mode
    model.eval()
    logging.info("Fine-tuned model loaded successfully!")
    
    # Load the test dataset
    logging.info("[2/3] Loading test dataset...")
    test_dataset = load_test_dataset(args.test_dataset, tokenizer)
    logging.info(f"Test dataset loaded with {len(test_dataset)} examples")
    
    # Calculate perplexity
    logging.info("[3/3] Calculating perplexity for fine-tuned model...")
    perplexity = ppl_model(model, tokenizer, test_dataset)
    
    # Add results to comparison
    model_name = args.model_dir.split('/')[-2] if '/' in args.model_dir else args.model_dir
    add_to_comparison(f"Fine-tuned: {model_name}", perplexity)
    
    logging.info(f"Fine-tuned model perplexity: {perplexity.item():.4f}")
    
    # Print comparison if multiple models have been evaluated
    logging.info("="*80)
    logging.info("FINAL RESULTS")
    logging.info("="*80)
    print_model_comparison()
    
    # Save results to file
    output_file = f"{args.model_dir}/perplexity_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERPLEXITY RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Fine-tuned Model: {model_name}\n")
        f.write(f"Model Directory: {args.model_dir}\n")
        f.write(f"Test Dataset: {args.test_dataset}\n")
        f.write(f"Number of examples: {len(test_dataset)}\n")
        f.write(f"\nFine-tuned Model Perplexity: {perplexity.item():.4f}\n")
        if args.compare_base:
            f.write(f"Base Model Perplexity: {base_perplexity.item():.4f}\n")
            improvement = ((base_perplexity.item() - perplexity.item()) / base_perplexity.item()) * 100
            f.write(f"\nImprovement: {improvement:.2f}%\n")
        f.write("\n" + "="*80 + "\n")
    
    logging.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
