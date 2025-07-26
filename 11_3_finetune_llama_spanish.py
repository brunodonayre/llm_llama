#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install torch transformers datasets accelerate peft bitsandbytes sentencepiece')


# # Ir a https://www.llama.com/llama-downloads/ para registrarse a Llama para descargar el modelo mas conveniente

# In[4]:


get_ipython().system('pip install llama-stack')


# In[5]:


get_ipython().system('pip install llama-stack -U')


# In[3]:


get_ipython().system('llama model list')


# In[6]:


get_ipython().system('llama model list --show-all')


# In[7]:


get_ipython().system('llama model download --source meta --model-id  Llama3.2-3B --meta-url "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaXQ2cHBoeHpna2Fhejhxa3h0OGk0Z3VpIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MTkzOTUzMn19fV19&Signature=YSB6umuYUOsj559VPVKW8349wsUWSvsILQMAFtApGfEs5sfULp7ui98TfaNmEe7SD7B%7EaVrLcqmeS-SUMN07kpa8ldk97Om4yOMtKzLXX%7Em3escyy3VPGUNoZQl4haZyujtyb8NA-Ln6w4CBUluln6qKHgU6aPCDvkDJND4gugyRFsnOfFunVuh02cw6sBRWDzjyoz0NeJUqs7dC%7EHlsdve3a7-yHqJeeOIRGoMDRK%7E2tXLoy8WlRpX5OJjTZTsM2DFOM7k8PrlBV2pJGkMzWu-TgmiNHkVw217ErpYKhBXYEnSy9K9-AIwVyMygk%7EdOrRZ%7Ese60pLvwf2CnA5cQcw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=648992861487744" --insecure')


# In[ ]:


get_ipython().system('llama model download --source meta --model-id  Llama3.2-3B')


# In[2]:


get_ipython().system('pip install huggingface-hub')


# In[1]:


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
import os

# Authenticate with Hugging Face
from huggingface_hub import login
login(token="TOKEN")  # Replace with your token

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"  # Adjust based on the exact LLaMA 3.1 model name
DATASET_PATH = "../Datos/spanish_chatbot_dataset.json"
OUTPUT_DIR = "llama3-spanish-chatbot"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization (4-bit + CPU offload)
# The BitsAndBytesConfig is used for quantizing large language models (LLMs) to make them more memory-efficient 
# during training and inference. Below is a detailed breakdown of each hyperparameter in your configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Loads the model in 4-bit precision instead of the usual 32-bit (FP32) or 16-bit (FP16/BF16). This reduces memory usage by ~8x compared to FP32.
    bnb_4bit_quant_type="nf4", # Specifies the 4-bit quantization algorithm. Here, "nf4" stands for NormalFloat 4-bit, an optimized format for normally distributed weights 
    bnb_4bit_compute_dtype=torch.float16, # Sets the compute dtype for operations (e.g., matrix multiplications) to FP16. While weights are stored in 4-bit, computations happen in 16-bit for better numerical stability.
    bnb_4bit_use_double_quant=True, # Enables double quantization, which quantizes the quantization constants (saving even more memory).
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# LoRA (Low-Rank Adaptation) Config
# LoRA is a parameter-efficient fine-tuning (PEFT) method that freezes the original large language model (LLM) and 
# injects small trainable low-rank matrices into specific layers. This allows fine-tuning with far fewer parameters
# (often <1% of the original model size) while maintaining performance.

peft_config = LoraConfig(
    r=16, # Rank of the low-rank matrices
    lora_alpha=32, # Scaling factor for LoRA weights
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Whether to train bias parameters
    task_type="CAUSAL_LM",  # Task type (causal language modeling)
    target_modules=["q_proj", "v_proj"]  # Modules to apply LoRA to: ["q_proj", "v_proj"] applies LoRA to the query and value projections in transformer attention layers.
)

# Training arguments
training_args = TrainingArguments(
    # Basic Training Setup
    output_dir=OUTPUT_DIR, #Directory where the trained model checkpoints and logs will be saved.
    num_train_epochs=3, # Number of full passes through the dataset (3 epochs here).
    per_device_train_batch_size=4, # Batch size per GPU/CPU. Higher values require more memory.
    gradient_accumulation_steps=2, # Splits a large batch into smaller chunks to save memory.
    # Optimization & Learning Rate
    optim="paged_adamw_32bit", # Uses a memory-efficient version of AdamW optimizer (good for large models like Llama).
    learning_rate=2e-4, # Initial learning rate. A common starting point for fine-tuning LLMs.
    weight_decay=0.001, # L2 regularization to prevent overfitting (penalizes large weights).  
    fp16=True, # Uses mixed-precision training (16-bit floats) to reduce memory usage and speed up training.
    bf16=False, # Disables BFloat16 precision (use bf16=True if your hardware supports it, e.g., modern GPUs/TPUs).
    max_grad_norm=0.3, # Clips gradients to prevent exploding gradients (limits the norm to 0.3).
    warmup_ratio=0.03, # Gradually increases LR from 0 to 2e-4 over 3% of training steps (avoids early instability).
    lr_scheduler_type="cosine", # Learning rate schedule: smoothly decreases LR in a cosine curve for better convergence.
    # Efficiency & Logging
    save_steps=500, # Saves a model checkpoint every 500 steps.
    group_by_length=True, # Groups similar-length sequences together to minimize padding and improve efficiency.
    logging_steps=10,  # Logs training metrics (loss, LR, etc.) every 10 steps.
    report_to="tensorboard", # Logs metrics to TensorBoard for visualization (alternatives: "wandb", "mlflow").
    # Evaluation
    evaluation_strategy="no", # Disables evaluation during training (set to "steps" or "epoch" if you have a validation set).
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="output",  # Using 'output' as our training text
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Train
trainer.train()

# Save model
trainer.model.save_pretrained(OUTPUT_DIR)

print("Training complete! Model saved to:", OUTPUT_DIR)


# In[ ]:


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel, PeftConfig

# Configuration
PEFT_MODEL_ID = "llama3-spanish-chatbot"  # Path to your saved model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base model and tokenizer
base_model = "meta-llama/Llama-3.2-3B"  # Original base model you fine-tuned from
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned PEFT model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, PEFT_MODEL_ID)
model = model.merge_and_unload()  # Merge LoRA adapters with base model

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE
)

# Test cases
test_prompts = [
    "¿Cuál es la capital de España?",  # What is the capital of Spain?
    "Explícame el concepto de inteligencia artificial",  # Explain AI
    "Recomiéndame un libro interesante",  # Recommend a book
    "¿Cómo puedo aprender programación?",  # How to learn programming
    "Háblame sobre la historia de México"  # Tell me about Mexico's history
]

# Generate responses
for prompt in test_prompts:
    print(f"\n=== Prompt: {prompt} ===")

    # Generate response
    output = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    print("Response:", output[0]['generated_text'])

    # Optional: Calculate perplexity (measure of model confidence)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        perplexity = torch.exp(outputs.loss).item()
        print(f"Perplexity: {perplexity:.2f}")

# Optional: Interactive chat mode
print("\n=== Interactive Mode (type 'quit' to exit) ===")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    output = pipe(
        user_input,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    print("Bot:", output[0]['generated_text'])

