"""
Download the Mistral model for offline use.
Run this script when you have internet access to cache the model locally.
"""

from unsloth import FastLanguageModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model configuration (must match fine-tune_mistral.py)
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True

logger.info("="*80)
logger.info("DOWNLOADING MODEL FOR OFFLINE USE")
logger.info("="*80)
logger.info(f"Model: {model_name}")
logger.info(f"Max sequence length: {max_seq_length}")
logger.info(f"This may take several minutes depending on your internet speed...")
logger.info("="*80 + "\n")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ MODEL DOWNLOADED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("The model has been cached in: ~/.cache/huggingface/hub/")
    logger.info("You can now run fine-tune_mistral.py in offline mode.")
    logger.info("="*80 + "\n")
    
except Exception as e:
    logger.error("\n" + "="*80)
    logger.error("✗ DOWNLOAD FAILED!")
    logger.error("="*80)
    logger.error(f"Error: {e}")
    logger.error("\nPlease check:")
    logger.error("1. Your internet connection")
    logger.error("2. HuggingFace Hub access (try: huggingface-cli login)")
    logger.error("3. Sufficient disk space (~15GB required)")
    logger.error("="*80 + "\n")
    raise
