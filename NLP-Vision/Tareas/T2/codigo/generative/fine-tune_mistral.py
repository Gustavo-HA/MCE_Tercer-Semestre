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

max_seq_length = 2048
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
dtype=None
load_in_4bit=True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
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
    "train": "./data/train_lyrics_alpaca.json",
    "test": "./data/test_lyrics_alpaca.json",
}

dataset = load_dataset("json", data_files=split_map)

# Function to format Alpaca dataset for chat template
def format_alpaca_to_chat(example):
    """
    Convert Alpaca format (instruction, input, output) to chat format
    for tokenizer.apply_chat_template
    """
    # Combine instruction and input into the user message
    user_message = example["instruction"]
    if example["input"]:
        user_message += f"\n\n{example['input']}"
    
    # Create conversation format
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

# Apply formatting to dataset
dataset = dataset.map(format_alpaca_to_chat, remove_columns=["instruction", "input", "output"])

# Train the model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 2, # For testing purposes. Use max_steps or num_train_epochs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
print(trainer_stats.metrics)