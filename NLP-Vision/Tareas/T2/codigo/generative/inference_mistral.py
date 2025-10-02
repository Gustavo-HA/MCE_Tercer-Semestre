import logging
import os
import argparse

# Set offline mode BEFORE importing any HuggingFace libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from unsloth import FastLanguageModel
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate lyrics with fine-tuned Mistral model')
parser.add_argument('--model_dir', type=str, required=True, help='Path to fine-tuned model directory')
parser.add_argument('--artist', type=str, default='Kendrick Lamar', help='Artist style (default: Kendrick Lamar)')
parser.add_argument('--prompt', type=str, default=None, help='Custom prompt/instruction')
parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate (default: 512)')
parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (default: 0.7)')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling (default: 0.9)')
parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty (default: 1.1)')

args = parser.parse_args()

logger.info("="*80)
logger.info("INFERENCE CONFIGURATION")
logger.info("="*80)
logger.info(f"Model directory: {args.model_dir}")
logger.info(f"Artist style: {args.artist}")
logger.info(f"Max tokens: {args.max_tokens}")
logger.info(f"Temperature: {args.temperature}")
logger.info(f"Top-p: {args.top_p}")
logger.info(f"Repetition penalty: {args.repetition_penalty}")
logger.info("="*80 + "\n")

# Check if model directory exists
if not os.path.exists(args.model_dir):
    logger.error(f"Model directory not found: {args.model_dir}")
    logger.error("Available models:")
    models_dir = "models"
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                final_model = os.path.join(item_path, "final_model")
                if os.path.exists(final_model):
                    logger.error(f"  - {final_model}")
    exit(1)

# Load the model
logger.info("Loading fine-tuned model...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    logger.info("✓ Model loaded successfully!\n")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

# Prepare model for inference
FastLanguageModel.for_inference(model)

# Create the prompt
if args.prompt:
    instruction = args.prompt
else:
    instruction = f"Write hip-hop lyrics in the style of {args.artist}"

# Format with Alpaca template
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Prepare input
inputs = tokenizer(
    [alpaca_prompt.format(instruction, "")],
    return_tensors="pt"
).to("cuda" if torch.cuda.is_available() else "cpu")

logger.info("="*80)
logger.info("PROMPT")
logger.info("="*80)
logger.info(instruction)
logger.info("="*80 + "\n")

logger.info("Generating lyrics...\n")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    repetition_penalty=args.repetition_penalty,
    use_cache=True,
)

# Decode output
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the response part (after "### Response:")
if "### Response:" in generated_text:
    response = generated_text.split("### Response:")[1].strip()
else:
    response = generated_text

logger.info("="*80)
logger.info("GENERATED LYRICS")
logger.info("="*80)
print(response)
logger.info("="*80)

# Save to file
output_file = f"generated_lyrics_{args.artist.replace(' ', '_').lower()}.txt"
with open(output_file, 'w') as f:
    f.write(f"Artist Style: {args.artist}\n")
    f.write(f"Prompt: {instruction}\n")
    f.write("="*80 + "\n\n")
    f.write(response)
    f.write("\n")

logger.info(f"\n✓ Lyrics saved to: {output_file}")
