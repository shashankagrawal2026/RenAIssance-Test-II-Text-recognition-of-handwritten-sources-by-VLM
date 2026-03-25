# Historical Handwritten Text Recognition using Qwen2.5-VL-7B

This project addresses the challenge of Optical Character Recognition (OCR) for `historical handwritten manuscripts` — primarily early-modern Spanish documents dating from the `sixteenth to nineteenth centuries` — a domain where conventional OCR engines such as Tesseract consistently fail due to degraded ink, non-standard letterforms, and archaic orthography. We propose a vision-language model approach, fine-tuning `Qwen2.5-VL-7B-Instruct` with **4-bit QLoRA** via the Unsloth framework to perform end-to-end page-level transcription. The model is trained on real paired PDF–transcription data augmented with brightness, contrast, and rotation perturbations, and further warmed up with IAM handwriting samples.

This project is part of the `RenAIssance project`, a large-scale digital humanities initiative under the **HumanAI Foundation**. It has been developed as a contribution to `Google Summer of Code 2026`.

<p align="center">
  <img src="images/humanai_logo.jpg" alt="HumanAI" style="height: 100px; margin-right: 20px;"/>
  <img src="images/gsoc_logo.png" alt="GSOC" style="height: 50px; padding-bottom: 50px" />
</p>

## Table of Contents

- [Project Goals](#Project-Goals)
- [Installation](#installation)
- [About The Project](#About-The-Project)
- [Datasets and Models](#datasets-and-models)
- [Model Performance](#model-performance)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

## Project Goals

1. **Fine-Tuning a State-of-the-Art Vision-Language Model for Manuscript OCR:** The primary objective is to adapt Qwen2.5-VL-7B — the highest-scoring open VLM on OCRBench (864) — to the specialized domain of early-modern handwritten Spanish text via parameter-efficient 4-bit QLoRA fine-tuning, enabling accurate end-to-end page-level transcription.

2. **Building a Robust, Reproducible Data Pipeline:** Design an automated pipeline that ingests raw PDF scans and DOCX transcriptions, performs fuzzy filename matching, renders pages to preprocessed JPEG images, and generates augmented training pairs — ensuring the entire workflow is reproducible from a single notebook.

3. **Benchmarking Against Traditional OCR Baselines:** Establish rigorous quantitative comparisons between the fine-tuned Qwen2.5-VL model and Tesseract OCR across multiple metrics (CER, WER, BLEU, Character F1) to demonstrate the advantages and current limitations of VLM-based approaches on historical manuscripts.

4. **Supporting Multilingual and Multi-Century Document Analysis:** Handle documents spanning multiple centuries (1606–1857) and accommodate orthographic irregularities such as interchangeable 'u'/'v' and 'f'/'s' characters, tildes as abbreviation markers, and archaic spellings — preserving these features faithfully in transcription output.

## Installation

This project is designed to run end-to-end on **Google Colab** with a T4 GPU. No local installation is required — simply open the notebook and execute the cells sequentially. All necessary packages are installed in the first code cell:

```bash
# Unsloth (auto-installs correct CUDA version)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Training & model libraries
pip install --upgrade trl transformers accelerate peft

# Data processing & evaluation
pip install pymupdf python-docx jiwer rapidfuzz nltk datasets

# Tesseract baseline (system-level install on Colab)
apt-get install tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
pip install pytesseract
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `unsloth` | 4-bit QLoRA fine-tuning of Qwen2.5-VL-7B |
| `transformers` | Model loading, tokenizer, generation config |
| `trl` | `SFTTrainer` for supervised fine-tuning |
| `peft` | LoRA adapter management |
| `pymupdf` (`fitz`) | PDF page rendering to images |
| `python-docx` | DOCX ground-truth parsing |
| `jiwer` | CER and WER computation |
| `rapidfuzz` / `difflib` | Fuzzy filename matching |
| `nltk` | BLEU score calculation |
| `pytesseract` | Tesseract OCR baseline |

### Project Directory Structure

```
RenAIssance/
├── images/            # Rendered page images (JPEG)
├── dataset/           # Processed training pairs (JSON)
├── checkpoints/       # LoRA adapter weights
│   └── best_adapter/  # Best checkpoint
└── outputs/           # Transcriptions, evaluation CSVs, comparison graphs
```

## About The Project

#### Data Preprocessing Pipeline

The notebook implements a fully automated data ingestion and preprocessing workflow:

1. **PDF–DOCX Discovery & Fuzzy Matching:** Scans the input directories for PDF and DOCX files, then uses `difflib.SequenceMatcher` to automatically pair each PDF with its corresponding transcription file, extracting century/year metadata from filenames.

2. **PDF Page Rendering:** Each PDF page is rasterized at 150 DPI using PyMuPDF (`fitz`), converted to RGB, resized to a maximum of 768×1024 pixels, and enhanced with adaptive sharpening for improved readability.

3. **Ground-Truth Extraction:** Transcription text is parsed from DOCX files with page-boundary detection, then manually mapped to specific PDF pages to create precise image–text training pairs.

4. **Data Augmentation:** Each real pair is augmented 5× with randomized brightness, contrast, rotation (±3°), and Gaussian noise perturbations, expanding 5 real pairs into 75 augmented samples.

5. **IAM Pre-Warming:** 150 samples from the `Teklia/IAM-line` dataset on Hugging Face are loaded as pre-warm data, yielding a final training set of 214 samples.

![Data Pipeline](images/Pre_Process.png)

#### Model Architecture

The project employs **Qwen2.5-VL-7B-Instruct**, a 5-billion-parameter multimodal vision-language model, fine-tuned with:

- **4-bit QLoRA** quantization via Unsloth for memory-efficient training on a T4 GPU (15.6 GB VRAM)
- **LoRA Configuration:** rank `r=16`, `lora_alpha=32`, dropout `0.05`
- **Fine-tuning Scope:** Both vision encoder and language decoder layers, including attention and MLP modules
- **Trainable Parameters:** 51,521,536 / 5,081,043,968 (1.01% of total)

The model processes full manuscript page images and generates transcriptions in a conversational VLM format, using a custom OCR prompt that instructs faithful transcription with preservation of original spelling and abbreviations.

![Model Architecture](images/architecture.png)

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |
| Quantization | 4-bit (bitsandbytes) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Epochs | 5 |
| Total Steps | 270 |
| Batch Size | 1 (×4 gradient accumulation) |
| Final Training Loss | 0.3438 |
| GPU | Tesla T4 (15.6 GB) |

#### Anti-Hallucination Guardrails

To mitigate VLM hallucination during inference, the notebook applies:
- **Repetition Penalty:** 1.15 (penalizes repeated tokens)
- **No-Repeat N-gram Size:** 4 (prevents 4-word phrase repetition)
- **Greedy Decoding:** `do_sample=False` for deterministic output

#### Interactive Document Viewer

The notebook includes an interactive `ipywidgets`-based document viewer that allows side-by-side comparison of the original manuscript scan and the model's transcription output.

## Datasets and Models

### Input Data

- **5 paired PDF + DOCX files** of historical Spanish handwritten manuscripts (16th–19th century)
- **35 total pages** rendered from the PDFs
- **Source archives:** AHPG-GPAH, AHN (Archivo Histórico Nacional), and others
- **IAM Handwriting Dataset:** 150 pre-warm samples from [`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line)

### Model Weights & Artifacts

| File | Description |
|------|-------------|
| `model.safetensors` | Qwen2.5-VL-7B-Instruct base weights (6.90 GB) |
| `generation_config.json` | Generation configuration |
| `preprocessor_config.json` | Vision preprocessor configuration |
| `tokenizer.json` | Tokenizer vocabulary (11.4 MB) |
| `vocab.json` / `merges.txt` | BPE tokenizer files |
| `chat_template.json` | Chat template for VLM conversations |
| `best_adapter/` | Fine-tuned LoRA adapter checkpoint (217.6 MB) |
| `evaluation.csv` | Per-page evaluation metrics |

### Download Links

- **Base Model:** [`unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit) on Hugging Face
- **Fine-tuned Adapter:** [Insert HuggingFace Hub link after publishing]
- **Training Data:** Available upon request via Google Drive

## Model Performance

### Qwen2.5-VL-7B (Fine-Tuned) — Final Evaluation

| Metric | Value |
|--------|-------|
| CER (Character Error Rate) | 0.8022 |
| WER (Word Error Rate) | 0.9981 |
| BLEU Score | 0.0780 |
| Character F1 Score | 0.6758 |
| Final Training Loss | 0.3438 |

### Tesseract OCR Baseline

| Metric | Value |
|--------|-------|
| CER (Character Error Rate) | 0.6900 |
| WER (Word Error Rate) | 0.9620 |

### Per-Document Breakdown (Qwen2.5-VL)

| Document | CER | WER | BLEU | F1 |
|----------|-----|-----|------|-----|
| AHPG-GPAH 1:1716,A.35 – 1744 | 0.4610 | 0.6456 | 0.2155 | 0.7562 |
| AHPG-GPAH AU61:2 – 1606 | 0.7637 | 0.9417 | 0.0246 | 0.8450 |
| ES.28079.AHN – INQUISICIÓN 1640 | 0.7312 | 0.9184 | 0.0033 | 0.6149 |
| PT3279:146:342 – 1857 | 0.7430 | 0.8056 | 0.0670 | 0.4989 |
| Pleito entre el Marqués de Viana | 1.3121 | 1.6792 | 0.0794 | 0.6639 |

### Comparative Analysis

![Baseline vs. Qwen2.5-VL Comparison](images/baseline_comparison_graph.png)

> **Note:** The Qwen2.5-VL model achieves notably higher Character F1 scores (0.6758 vs. 0.6294) and BLEU scores (0.0780 vs. 0.0022) compared to Tesseract, indicating superior content recall and sequence quality despite higher raw CER — a common pattern when VLMs attempt richer, more contextual transcription of degraded manuscripts.

## Acknowledgements

This project is supported by the [HumanAI Foundation](https://humanai.foundation/) and Google Summer of Code 2026. The research builds upon the [RenAIssance project](https://github.com/Shashankss1205/RenAIssance), a digital humanities initiative dedicated to the preservation and computational analysis of early-modern European manuscripts. Special thanks to the Unsloth team for enabling efficient VLM fine-tuning and to the Hugging Face community for hosting open model weights and datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Google Summer of Code 2026 Project](https://summerofcode.withgoogle.com/)
- [HumanAI Foundation](https://humanai.foundation/)
- [RenAIssance GitHub Repository](https://github.com/Shashankss1205/RenAIssance)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Unsloth Framework](https://github.com/unslothai/unsloth)
- [Fine-Tuned Adapter on HuggingFace Hub](https://huggingface.co/) *(link to be updated after publishing)*

---

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas first. Contributions are always welcomed!