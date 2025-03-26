import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import load_dataset
import docx

def docx_to_text(file_path):
    """Converts a DOCX file to a plain text string."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def main():
    # ---------- 1. Convert DOCX to Text and Save as train.txt ----------
    docx_file = "Zelle.docx"
    txt_file = "train.txt"
    
    if not os.path.exists(txt_file):
        print(f"Converting {docx_file} to {txt_file}...")
        text = docx_to_text(docx_file)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        print("Conversion complete!")
    else:
        print(f"{txt_file} already exists. Skipping conversion.")
    
    # ---------- 2. Load the Dataset ----------
    dataset = load_dataset("text", data_files={"train": txt_file})
    
    # ---------- 3. Load GPT-2 Tokenizer and Model ----------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Set the padding token to be the same as the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    
    # ---------- 4. Tokenize the Dataset (with labels) ----------
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        # For causal language modeling, use input_ids as labels.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # ---------- 5. Set Up Data Collator and Training Arguments ----------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,                   # Adjust epochs as needed
        per_device_train_batch_size=2,        # Adjust based on available GPU/CPU memory
        gradient_accumulation_steps=8,        # Simulate a larger batch size if necessary
        evaluation_strategy="no",             # No evaluation during training
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
    )
    
    # ---------- 6. Initialize the Trainer and Fine-Tune ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")
    
    # (Optional) Save the fine-tuned model and tokenizer for later use
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    # ---------- 7. Generate Text from a Prompt (English) ----------
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define your English prompt (modify as needed)
    prompt = ("I looked through the window, but I saw nothing. "
              "Only the window cross that cut the blue sky into four rectangular pieces. "
              "They reminded me of")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        sample_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,             # Enable randomness
            max_length=150,             # Adjust max length as needed
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=3      # Generate three different outputs
        )
    
    english_outputs = []
    print("=== Generated English Texts ===")
    for i, sample in enumerate(sample_outputs):
        text = tokenizer.decode(sample, skip_special_tokens=True)
        english_outputs.append(text)
        print(f"--- English Output {i+1} ---")
        print(text)
        print("\n")
    
    # ---------- 8. Translate the Generated English Texts to German ----------
    translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
    print("=== Translated German Texts ===")
    for i, text in enumerate(english_outputs):
        translation = translator(text, max_length=400)
        german_text = translation[0]['translation_text']
        print(f"--- German Translation {i+1} ---")
        print(german_text)
        print("\n")

if __name__ == "__main__":
    main()
