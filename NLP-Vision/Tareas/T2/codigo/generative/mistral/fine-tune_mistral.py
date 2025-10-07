import logging
import os
from datetime import datetime
import argparse

# Offline para cluster Bajio
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from unsloth import (
    FastLanguageModel,
    get_chat_template,
)
import torch
from trl import (
    SFTConfig,
    SFTTrainer,
)
from datasets import load_dataset

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Parametros
parser = argparse.ArgumentParser(description='Fine-tune Mistral model with LoRA')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
parser.add_argument('--batch_size', type=int, default=2, help='Per-device batch size (default: 2)')
parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps (default: 8)')
parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank (default: 32)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (default: auto-generated)')
parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length (default: 2048)')
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
grad_accum = args.grad_accum
lora_r = args.lora_r
max_seq_length = args.max_seq_length

# Configuracion del modelo
dtype = None
load_in_4bit = True
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

logger.info("="*80)
logger.info("TRAINING CONFIGURATION")
logger.info("="*80)
logger.info(f"Learning rate: {lr}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Gradient accumulation: {grad_accum}")
logger.info(f"Effective batch size: {batch_size * grad_accum}")
logger.info(f"LoRA rank: {lora_r}")
logger.info(f"Max sequence length: {max_seq_length}")
logger.info("="*80 + "\n")

# Cargar el modelo de Unsloth en local con la cache de HuggingFace
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_cache_name = model_name.replace("/", "--")
model_cache_path = f"{cache_dir}/models--{model_cache_name}"

if os.path.exists(model_cache_path):
    refs_main = f"{model_cache_path}/refs/main"
    if os.path.exists(refs_main):
        with open(refs_main, 'r') as f:
            commit_hash = f.read().strip()
        model_path = f"{model_cache_path}/snapshots/{commit_hash}"
        logger.info(f"Found cached model at: {model_path}")
    else:
        logger.error(f"Cache exists but no 'main' ref found at: {refs_main}")
        model_path = model_name  # Volver al modo online
else:
    logger.warning("Descargando de HuggingFace Hub...")
    model_path = model_name

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    logger.info("Modelo cargado exitosamente!")
except Exception as e:
    logger.error(f"No se pudo cargar el modelo: {e}")
    raise

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_r,
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha=lora_r * 2,
    lora_dropout=0.00,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="alpaca",
)

split_map = {
    "train": "./data/text_gen/train_lyrics_alpaca.json",
    "test": "./data/text_gen/test_lyrics_alpaca.json",
}

dataset = load_dataset("json", data_files=split_map)


def format_alpaca_to_chat(example):
    """
    Convert Alpaca format (instruction, input, output) to chat format
    for tokenizer.apply_chat_template
    """

    user_message = example["instruction"]
    if example["input"]:
        user_message += f"\n\n{example['input']}"
    
    conversation = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

# Aplicar el formato Alpaca al datset
dataset = dataset.map(format_alpaca_to_chat, remove_columns=["instruction", "input", "output"])

# Generar nombre de directorio de salida
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.output_dir:
    output_dir = f"models/{args.output_dir}"
else:
    # Auto-generate directory name with hyperparameters
    run_name = f"mistral_lr{lr}_ep{epochs}_bs{batch_size}x{grad_accum}_r{lora_r}_{timestamp}"
    output_dir = f"models/{run_name}"

logger.info(f"Output directory: {output_dir}")

# Entrenar el modelo
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        warmup_steps = 5,
        #max_steps = 2, # For testing purposes. Use max_steps or num_train_epochs
        num_train_epochs= epochs,
        learning_rate = lr,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none", # Use this for WandB etc
    ),
)

num_gpus = torch.cuda.device_count()
logger.info(f"Number of available GPUs: {num_gpus}")

start_gpu_memory = {}
max_memory = {}

for device_id in range(num_gpus):
    gpu_stats = torch.cuda.get_device_properties(device_id)
    start_mem = round(torch.cuda.max_memory_reserved(device_id) / 1024 / 1024 / 1024, 3)
    max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    start_gpu_memory[device_id] = start_mem
    max_memory[device_id] = max_mem
    
    logger.info(f"GPU {device_id}: {gpu_stats.name}. Max memory = {max_mem} GB.")
    logger.info(f"GPU {device_id}: {start_mem} GB of memory reserved.")

trainer_stats = trainer.train()
logger.info(f"Training metrics: {trainer_stats.metrics}")

# Save the model - Try GGUF first, fall back to HuggingFace format if it fails
try:
    logger.info("Attempting to save model in GGUF format...")
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    logger.info("✓ Model saved successfully in GGUF format!")
except Exception as e:
    logger.warning(f"Failed to save in GGUF format: {e}")
    logger.info("Falling back to HuggingFace format (LoRA adapters)...")
    final_output_dir = f"{output_dir}/final_model"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"✓ Model saved successfully in HuggingFace format at: {final_output_dir}")
    logger.info("You can merge the LoRA adapters later or use them directly for inference.")

# Report GPU memory usage
logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
logger.info(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)

# Log memory stats for all GPUs
for device_id in range(num_gpus):
    used_memory = round(torch.cuda.max_memory_reserved(device_id) / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory[device_id], 3)
    used_percentage = round(used_memory / max_memory[device_id] * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory[device_id] * 100, 3)
    
    logger.info(f"GPU {device_id} - Peak reserved memory = {used_memory} GB.")
    logger.info(f"GPU {device_id} - Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"GPU {device_id} - Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"GPU {device_id} - Peak reserved memory for training % of max memory = {lora_percentage} %.")