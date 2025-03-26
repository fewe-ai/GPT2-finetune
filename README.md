# Fine-Tuning GPT-2 and Translating Output to German

This project provides a complete pipeline to:
- Convert a DOCX file (e.g., `Zelle.docx`) into a text file.
- Fine-tune the GPT-2 language model on the extracted text.
- Generate text continuations based on a given prompt.
- Translate the generated English text into German using a pre-trained translation model.

## Features

- **DOCX to Text Conversion:**  
  Automatically extracts and saves text from a DOCX file.

- **Dataset Preparation & Fine-Tuning:**  
  Loads text data from a file, tokenizes it, and fine-tunes GPT-2 with the Hugging Face Trainer.

- **Text Generation:**  
  Generates multiple text outputs from a provided English prompt.

- **Translation:**  
  Translates the generated English text into German using the Helsinki-NLP `opus-mt-en-de` model.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [python-docx](https://python-docx.readthedocs.io/en/latest/)
- [SentencePiece](https://github.com/google/sentencepiece) (for the translation model tokenizer)

Install the required packages using:

```bash
pip install torch transformers datasets python-docx sentencepiece
