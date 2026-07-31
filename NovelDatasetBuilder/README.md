# NovelDatasetBuilder

> **Automated pipeline** that converts a single Novel PDF into a clean, deduplicated, GPT-ready instruction-tuning dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [API Key Setup](#api-key-setup)
5. [How to Add a PDF](#how-to-add-a-pdf)
6. [How to Run](#how-to-run)
7. [How the Dataset is Generated](#how-the-dataset-is-generated)
8. [Output Format](#output-format)
9. [How to Fine-Tune with final_dataset.jsonl](#how-to-fine-tune-with-final_datasetjsonl)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

NovelDatasetBuilder takes **one PDF novel** and produces **one JSONL file** (`output/final_dataset.jsonl`) that is immediately ready for GPT fine-tuning.

### What it does

| Stage | Description |
|-------|-------------|
| 1 – Extract | Reads every page of the PDF; removes page numbers, headers, footers, broken OCR spacing |
| 2 – Detect | Finds chapters, scenes, acts, prologues, epilogues; falls back to paragraph clustering |
| 3 – Chunk | Creates 1 200–2 500-word intelligent chunks that never split dialogue or paragraphs |
| 4 – Generate | Calls the LLM for every chunk → 40 unique instruction-response pairs per chunk |
| 5 – Validate | Repairs malformed JSON; drops empty, broken, or too-short entries |
| 6 – Deduplicate | Removes exact and near-duplicate pairs using TF-IDF cosine similarity |

---

## Folder Structure

```
NovelDatasetBuilder/
│
├── input/
│      novel.pdf          ← your novel goes here
│
├── output/
│      final_dataset.jsonl  ← the only output file
│
├── src/
│      __init__.py
│      main.py            ← pipeline entry point
│      config.py          ← all configuration
│      extract_pdf.py     ← PDF reading & cleaning
│      detect_sections.py ← chapter/scene detection
│      chunk_builder.py   ← intelligent chunking
│      llm_generator.py   ← LLM calls & pair generation
│      validator.py       ← JSON validation & repair
│      deduplicate.py     ← deduplication
│      utils.py           ← shared helpers
│
├── prompts/
│      generation_prompt.txt  ← LLM master prompt
│
├── requirements.txt
├── README.md
└── .env                  ← your API key (never commit this)
```

---

## Installation

### Prerequisites

- Python **3.11** or higher
- pip

### Steps

```bash
# 1. Navigate into the project folder
cd NovelDatasetBuilder

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## API Key Setup

1. Open (or create) the `.env` file in the project root.
2. Set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-key-here
```

3. (Optional) Change the model:

```env
LLM_MODEL=gpt-4o          # best quality
# LLM_MODEL=gpt-4o-mini   # faster and cheaper
```

> **Important**: Never commit your `.env` file to any version control system.

---

## How to Add a PDF

1. Place your novel PDF inside the `input/` folder.
2. Rename it to exactly `novel.pdf`.

```
NovelDatasetBuilder/
└── input/
       novel.pdf    ← rename your file to this
```

> **Note**: The PDF must contain selectable text (not a scanned image). If your PDF is scanned, run it through an OCR tool first (e.g., Adobe Acrobat, Tesseract, or AWS Textract).

---

## How to Run

From the **project root** directory (`NovelDatasetBuilder/`):

```bash
python -m src.main
```

Or equivalently:

```bash
python src/main.py
```

### What you will see

```
════════════════════════════════════════════════════════════════════════
  Stage 1 │ Extracting PDF text
════════════════════════════════════════════════════════════════════════
Extracted 87,342 words from the novel.

════════════════════════════════════════════════════════════════════════
  Stage 2 │ Detecting novel sections
════════════════════════════════════════════════════════════════════════
Detected 24 sections.

...

Generating dataset: 100%|████████████| 42/42 [chunks]   pairs: 1,680

════════════════════════════════════════════════════════════════════════
  Pipeline Complete
════════════════════════════════════════════════════════════════════════
  ✓ Input PDF          : input/novel.pdf
  ✓ Sections detected  : 24
  ✓ Chunks processed   : 42
  ✓ Pairs generated    : 1,680
  ✓ Pairs after valid  : 1,643
  ✓ Final dataset size : 1,589 pairs
  ✓ Output file        : output/final_dataset.jsonl
  ✓ Time elapsed       : 487.3 seconds
```

---

## How the Dataset is Generated

For every text chunk the pipeline sends a detailed prompt to the LLM asking it to generate **40 unique instruction-response pairs** covering:

| Category | Examples |
|----------|---------|
| Characters | traits, comparison, relationships, motivation |
| Themes | identification, comparison, symbolism |
| Literary Analysis | devices, conflict, author intention |
| Plot | summary, timeline, cause & effect |
| Exam Prep | 1/2/3/5/10/15-mark Q&A, MCQ, True/False, Fill-in |
| Study Tools | flashcards, revision notes, memory tricks |
| Higher Order | HOTS, critical thinking, viva, interview questions |
| Conversations | student-professor, misconception correction |

### Strict quality rules

- **No hallucinations** – responses are based only on the uploaded PDF.
- **No copy-paste** – every instruction and response is unique.
- **No conversation format** – instructions contain only the question; responses contain only the answer.
- **Professional English** – responses sound like an experienced university professor.

---

## Output Format

Each line of `output/final_dataset.jsonl` is a valid JSON object:

```json
{"instruction":"What is the central conflict in this chapter?","response":"The central conflict revolves around..."}
{"instruction":"Explain the symbolism of the river in this passage.","response":"The river serves as a powerful symbol of..."}
```

- **No markdown**
- **No comments**
- **No numbering**
- **One JSON object per line**
- **UTF-8 encoded**

---

## How to Fine-Tune with final_dataset.jsonl

### Using the OpenAI Fine-Tuning API

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-key")

# Step 1: Upload the dataset
with open("output/final_dataset.jsonl", "rb") as f:
    file_response = client.files.create(file=f, purpose="fine-tune")

file_id = file_response.id
print(f"Uploaded file ID: {file_id}")

# Step 2: Create a fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",   # or gpt-3.5-turbo-1106
)
print(f"Fine-tuning job: {job.id}")
```

> **Note**: OpenAI's fine-tuning API expects the `messages` format. Convert your dataset using the script below if needed.

### Converting to OpenAI Chat Format (if required)

```python
import json

with open("output/final_dataset.jsonl") as fin, \
     open("output/final_dataset_chat.jsonl", "w") as fout:
    for line in fin:
        pair = json.loads(line)
        chat_entry = {
            "messages": [
                {"role": "system", "content": "You are an expert literature professor."},
                {"role": "user",      "content": pair["instruction"]},
                {"role": "assistant", "content": pair["response"]},
            ]
        }
        fout.write(json.dumps(chat_entry) + "\n")
```

---

## Configuration Reference

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Model for generation |
| `LLM_TEMPERATURE` | `0.85` | Sampling temperature |
| `LLM_MAX_TOKENS` | `8000` | Max tokens per response |
| `LLM_MAX_RETRIES` | `5` | API retry attempts |
| `LLM_RETRY_DELAY` | `3.0` | Base retry delay (seconds) |
| `CHUNK_MIN_WORDS` | `1200` | Minimum words per chunk |
| `CHUNK_MAX_WORDS` | `2500` | Maximum words per chunk |
| `CHUNK_OVERLAP_WORDS` | `150` | Context overlap between chunks |
| `PAIRS_PER_CHUNK` | `40` | Pairs generated per chunk |
| `MIN_RESPONSE_WORDS` | `20` | Minimum response word count |
| `SIMILARITY_THRESHOLD` | `0.85` | Near-duplicate detection threshold |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY is not set` | Add your API key to `.env` |
| `Input PDF not found` | Ensure the file is at `input/novel.pdf` |
| `Extracted text is too short` | PDF may be scanned; run OCR first |
| `LLM returned empty response` | Check API quota; the retry logic will handle transient failures |
| Pipeline runs but dataset is tiny | Increase `PAIRS_PER_CHUNK` or lower `SIMILARITY_THRESHOLD` |
| Rate limit errors | Reduce `PAIRS_PER_CHUNK` or add delays via `LLM_RETRY_DELAY` |

---

## License

This project is for personal/research use. Always comply with the terms of service of any LLM API you use.
