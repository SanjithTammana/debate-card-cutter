# Debate Card Cutter

Experiments in turning debate evidence into formatted cards, plus a local MCP server for searching previously cut cards, parsing Verbatim DOCX files, and rendering marked cards back to DOCX.

## What runs today

`server.py` exposes four MCP tools:

| Tool | Function |
| --- | --- |
| `cut_card` | Retrieves similar human-formatted cards and returns examples plus instructions for a calling assistant. The MCP server does not itself generate a cut. |
| `search_cards` | Searches the local FAISS card index. |
| `parse_docx` | Extracts cards and formatting from a Verbatim-style DOCX file. |
| `render_card_docx` | Writes tagged text as a formatted DOCX card. |

The server uses `sentence-transformers/all-MiniLM-L6-v2` by default. The index must have been built with the same embedding model. It runs locally and does not call a model API; an assistant using `cut_card` still consumes its normal conversation tokens.

## Setup

Use Python 3.10 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

Build or supply a compatible `cards.index` and `cards_metadata.pkl` using the RAG notebook, then set:

```text
CARD_INDEX_PATH=/absolute/path/to/cards.index
CARD_METADATA_PATH=/absolute/path/to/cards_metadata.pkl
CARD_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # optional
```

For Claude Code, register the server with the corresponding paths on your machine:

```bash
claude mcp add debate-card-cutter \
  -e CARD_INDEX_PATH=/absolute/path/to/cards.index \
  -e CARD_METADATA_PATH=/absolute/path/to/cards_metadata.pkl \
  -- python /absolute/path/to/server.py
```

Only load `cards_metadata.pkl` from a source you trust: Python pickle files can execute code when loaded.

## Research notebooks

The notebooks are independent experiments, not a single end-to-end training pipeline. Their names describe the main approach; they do not imply that a model or dataset is shipped here.

| Notebook | Approach |
| --- | --- |
| `preprocessing_fast.ipynb` | Parses and aligns formatted cards from DOCX into training examples. |
| `rag_gemini.ipynb` | Retrieves examples with FAISS and prompts Gemini to format a new card. |
| `rag_groq.ipynb` | Groq variant of the retrieval and generation experiment. |
| `qwen_span_finetune.ipynb` | Fine-tunes Qwen 2.5 14B for span-oriented card formatting. |
| `qwen_inference_docx.ipynb` | Runs a trained Qwen model and exports a DOCX result. |
| `gptoss_span_experiment.ipynb` | GPT-OSS 20B formatting experiment. |
| `bert_span_baseline.ipynb` | Earlier token/span-model baseline. |
| `early_gpu_parser.ipynb` | Earlier parallel DOCX parsing and deduplication experiment. |

Some notebooks expect Google Colab, Google Drive paths, GPU hardware, external model downloads, or provider API keys. Supply those through your own Colab environment; no credentials are included. Notebook outputs and execution history were removed for this source release.

## Data and limits

The [public data package](data/README.md) contains a text-free formatting profile derived from 4,869 locally parsed cards, a synthetic example of the card schema, and a script to regenerate the profile from a compatible JSON file. It supports analysis of span frequency and data quality, but cannot run retrieval or train a text model by itself.

The full evidence corpus, per-card text and spans, FAISS index, pickled metadata, trained weights, and DOCX files are not included. The local source collection appears to include OpenCaselist disclosure files and third-party publications. Public access to those files does not establish permission to republish them. You need to create or provide data you are allowed to use to run retrieval or training.

This repository records research prototypes. It does not establish that a generated card is faithful to its source or that a tagline is supported by the highlighted words. Human review is required before using a cut in a round.
