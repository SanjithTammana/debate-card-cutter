"""
Debate Card Cutter — MCP Server (No API Required)
===================================================
Retrieval-only MCP server for Claude Code / Cowork.

The FAISS index handles finding similar human-formatted cards.
Claude (your subscription conversation) handles the actual formatting.
No API keys needed.

Tools:
  1. cut_card        — retrieve similar examples + return them with the system prompt
                       so Claude can format the card directly in conversation
  2. search_cards    — find similar cards in the FAISS index
  3. parse_docx      — extract cards from a Verbatim-formatted .docx file
  4. render_card_docx— render tagged text to a formatted .docx file

Setup:
  pip install "mcp[cli]" sentence-transformers faiss-cpu python-docx numpy

  Environment variables:
    CARD_INDEX_PATH    — path to cards.index
    CARD_METADATA_PATH — path to cards_metadata.pkl

  Claude Code:
    claude mcp add debate-card-cutter \
      -e CARD_INDEX_PATH="/path/to/cards.index" \
      -e CARD_METADATA_PATH="/path/to/cards_metadata.pkl" \
      -- python server.py
"""

import os
import re
import json
import html
import pickle
import hashlib
import logging
import zipfile
import difflib
from pathlib import Path
from typing import List, Dict, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("debate-card-cutter")

logger = logging.getLogger("debate-card-cutter")
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

EMBED_MODEL_NAME = os.getenv("CARD_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CARD_INDEX_PATH = os.getenv("CARD_INDEX_PATH", "")
CARD_METADATA_PATH = os.getenv("CARD_METADATA_PATH", "")

TOP_K = 2
RETRIEVAL_TEXT_CHARS = 2200
MAX_EXAMPLE_OUTPUT_CHARS = 1500

# ══════════════════════════════════════════════════════════════════════
# LAZY-LOADED GLOBALS
# ══════════════════════════════════════════════════════════════════════

_embedder = None
_faiss_index = None
_work_examples = None


def _ensure_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model ready.")
    return _embedder


def _ensure_faiss():
    global _faiss_index, _work_examples
    if _faiss_index is not None:
        return _faiss_index, _work_examples

    import faiss

    if not CARD_INDEX_PATH or not CARD_METADATA_PATH:
        raise RuntimeError(
            "FAISS index not configured.\n"
            "Set CARD_INDEX_PATH and CARD_METADATA_PATH env vars pointing to\n"
            "the cards.index and cards_metadata.pkl files from your Colab pipeline."
        )

    logger.info(f"Loading FAISS index from {CARD_INDEX_PATH}")
    _faiss_index = faiss.read_index(CARD_INDEX_PATH)

    logger.info(f"Loading metadata from {CARD_METADATA_PATH}")
    with open(CARD_METADATA_PATH, "rb") as f:
        _work_examples = pickle.load(f)

    logger.info(f"Loaded: {_faiss_index.ntotal} vectors, {len(_work_examples)} examples")
    return _faiss_index, _work_examples


# ══════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════

TAG_RE = re.compile(r"</?(ul|h|e)>")
ALLOWED_TAG_PATTERN = re.compile(r"</?(?:ul|h|e)>")


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def repair_broken_inline_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(
        r'(?<![A-Za-z0-9_])(ul|h|e)([^<>\n]{1,400}?)/\1(?![A-Za-z0-9_])',
        lambda m: f"<{m.group(1)}>{m.group(2)}</{m.group(1)}>",
        text,
    )
    text = re.sub(
        r'(?<![A-Za-z0-9_])(ul|h|e)\s+([^<>\n]{1,400}?)/\1(?![A-Za-z0-9_])',
        lambda m: f"<{m.group(1)}>{m.group(2)}</{m.group(1)}>",
        text,
    )
    return text


def clean_tagged_text(tagged_text: str) -> str:
    tagged_text = tagged_text.strip()
    tagged_text = html.unescape(tagged_text)
    tagged_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", tagged_text, flags=re.DOTALL).strip()
    tagged_text = repair_broken_inline_tags(tagged_text)

    def repl(match):
        token = match.group(0)
        return token if ALLOWED_TAG_PATTERN.fullmatch(token) else ""

    tagged_text = re.sub(r"<[^>]*>", repl, tagged_text)
    return tagged_text


# ══════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════

def retrieve_examples(new_tag: str, new_cite: str, new_body: str, top_k: int = TOP_K) -> List[Dict]:
    embedder = _ensure_embedder()
    index, work_examples = _ensure_faiss()

    query = f"TAG: {new_tag}\nCITE: {new_cite}\nBODY:\n{new_body[:RETRIEVAL_TEXT_CHARS]}"
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q, top_k * 8)

    retrieved = []
    seen_ids = set()
    for score, idx in zip(scores[0], idxs[0]):
        ex = work_examples[int(idx)]
        if ex["id"] in seen_ids:
            continue
        seen_ids.add(ex["id"])
        retrieved.append({**ex, "score": float(score)})
        if len(retrieved) >= top_k:
            break

    return retrieved


# ══════════════════════════════════════════════════════════════════════
# THE SYSTEM PROMPT (returned to Claude so it knows how to format)
# ══════════════════════════════════════════════════════════════════════

FORMATTING_INSTRUCTIONS = """You format competitive debate evidence cards.

Your task is to add inline formatting tags to the BODY ONLY.

Allowed tags:
- <ul>...</ul>  for underlined text
- <h>...</h>    for highlighted text
- <e>...</e>    for emphasized/bold text

Hard structural rules:
1. Every highlighted span must be fully inside an underlined span.
2. Every emphasized span must be fully inside a highlighted span.
3. Therefore, emphasized text must also be inside an underlined span.
4. Never produce a <h> span that is not nested inside <ul>.
5. Never produce a <e> span that is not nested inside <h> and <ul>.

Content rules:
6. Preserve the original body text exactly. Do not paraphrase, delete, add, or reorder words.
7. tagged_text must be the original body text plus only the allowed tags.
8. Do not tag the cite. Only tag the body text.
9. Return only the tagged body text.

Formatting density guidelines:
10. Underline (<ul>) the key evidentiary sentences throughout the card — the claims, warrants, causal links, and impact statements a debater would read aloud. Skip background, transitions, and setup sentences.
11. Highlight (<h>) the most important sub-phrase within each underlined span — the specific term, causal claim, or conclusion that carries the argument. Never highlight filler words like "the", "a", "is", "of", "and", "that", "which".
12. Emphasize (<e>) only the single most critical word inside a highlight. Use at most once or twice per card total — most highlights should have no <e>.
13. Do not underline entire paragraphs. Each <ul> span should be one sentence or a tight phrase, not multiple sentences.

Nesting examples:
- Valid: <ul>important <h>key <e>word</e></h> claim</ul>
- Invalid: <h>highlight without underline</h>
- Invalid: <e>emphasis outside highlight</e>"""


# ══════════════════════════════════════════════════════════════════════
# SPAN CONVERSION
# ══════════════════════════════════════════════════════════════════════

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted([[int(s), int(e)] for s, e in intervals if e > s], key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def subtract_intervals(base, covered):
    s, e = base
    if e <= s:
        return []
    covered = merge_intervals([[max(s, cs), min(e, ce)] for cs, ce in covered if ce > s and cs < e])
    if not covered:
        return [[s, e]]
    out = []
    cur = s
    for cs, ce in covered:
        if cur < cs:
            out.append([cur, cs])
        cur = max(cur, ce)
    if cur < e:
        out.append([cur, e])
    return out


def ensure_nested_spans(spans):
    ul = [[s, e] for s, e, t in spans if t == "ul" and e > s]
    h = [[s, e] for s, e, t in spans if t == "h" and e > s]
    e_ = [[s, e] for s, e, t in spans if t == "e" and e > s]

    ul, h, e_ = merge_intervals(ul), merge_intervals(h), merge_intervals(e_)

    extra_h = []
    for es, ee in e_:
        extra_h.extend(subtract_intervals([es, ee], h))
    h = merge_intervals(h + extra_h)

    extra_ul = []
    for hs, he in h:
        extra_ul.extend(subtract_intervals([hs, he], ul))
    ul = merge_intervals(ul + extra_ul)

    repaired = [[s, e, "ul"] for s, e in ul] + [[s, e, "h"] for s, e in h] + [[s, e, "e"] for s, e in e_]
    repaired.sort(key=lambda x: (x[0], x[1], x[2]))
    return repaired


def tagged_text_to_spans(tagged_text: str):
    spans = []
    plain_chars = []
    stack = []
    i = 0

    while i < len(tagged_text):
        matched = False
        for tag_name, open_tag, close_tag in [("ul", "<ul>", "</ul>"), ("h", "<h>", "</h>"), ("e", "<e>", "</e>")]:
            if tagged_text[i:].startswith(open_tag):
                stack.append((tag_name, len(plain_chars)))
                i += len(open_tag)
                matched = True
                break
            if tagged_text[i:].startswith(close_tag):
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j][0] == tag_name:
                        tag, start = stack.pop(j)
                        spans.append([start, len(plain_chars), tag])
                        break
                i += len(close_tag)
                matched = True
                break
        if not matched:
            plain_chars.append(tagged_text[i])
            i += 1

    for tag_name, start in stack:
        spans.append([start, len(plain_chars), tag_name])

    return "".join(plain_chars), ensure_nested_spans(spans)


# ══════════════════════════════════════════════════════════════════════
# DOCX PARSING (from preprocessing notebook)
# ══════════════════════════════════════════════════════════════════════

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = {"w": W_NS}
CARD_START_STYLE_NAMES = {"heading 4"}
STOP_STYLE_NAMES = {"heading 1", "heading 2", "heading 3", "heading 4"}


def qn(tag: str) -> str:
    _, local = tag.split(":")
    return f"{{{W_NS}}}{local}"


def normalize_style_name(style_name: str) -> str:
    return " ".join((style_name or "").strip().lower().split())


def parse_styles_xml(styles_bytes: bytes):
    sid2name, sid2based = {}, {}
    if not styles_bytes:
        return sid2name, sid2based
    root = ET.fromstring(styles_bytes)
    for style in root.findall("w:style", XML_NS):
        style_id = style.get(qn("w:styleId"))
        if not style_id:
            continue
        name_el = style.find("w:name", XML_NS)
        based_on_el = style.find("w:basedOn", XML_NS)
        sid2name[style_id] = name_el.get(qn("w:val"), style_id) if name_el is not None else style_id
        if based_on_el is not None:
            base_id = based_on_el.get(qn("w:val"))
            if base_id:
                sid2based[style_id] = base_id
    return sid2name, sid2based


def resolve_style_name(style_id, sid2name, sid2based, max_depth=10):
    if not style_id:
        return ""
    cur, seen = style_id, set()
    for _ in range(max_depth):
        if cur in seen:
            break
        seen.add(cur)
        name = sid2name.get(cur)
        if name:
            return name
        cur = sid2based.get(cur)
        if not cur:
            break
    return sid2name.get(style_id, style_id or "")


def extract_text_from_run_xml(run_el):
    return "".join(t.text or "" for t in run_el.findall(".//w:t", XML_NS))


def get_run_format_state_xml(run_el, sid2name, sid2based):
    rpr = run_el.find("w:rPr", XML_NS)
    is_ul = is_hl = is_em = False
    if rpr is not None:
        if rpr.find("w:u", XML_NS) is not None:
            is_ul = True
        if rpr.find("w:b", XML_NS) is not None:
            is_em = True
        hl = rpr.find("w:highlight", XML_NS)
        if hl is not None:
            val = hl.get(qn("w:val"))
            if val and val.lower() not in {"none", "auto"}:
                is_hl = True
        rstyle = rpr.find("w:rStyle", XML_NS)
        if rstyle is not None:
            sid = rstyle.get(qn("w:val"))
            if "Emphasis" in resolve_style_name(sid, sid2name, sid2based):
                is_em = True
    return is_ul, is_hl, is_em


def paragraph_style_name_xml(p_el, sid2name, sid2based):
    ppr = p_el.find("w:pPr", XML_NS)
    if ppr is None:
        return ""
    pstyle = ppr.find("w:pStyle", XML_NS)
    if pstyle is None:
        return ""
    return resolve_style_name(pstyle.get(qn("w:val")), sid2name, sid2based)


def paragraph_text_xml(p_el):
    return "".join(t.text or "" for t in p_el.findall(".//w:t", XML_NS)).strip()


def render_paragraph_from_xml(p_el, sid2name, sid2based):
    plain_parts, tagged_parts = [], []
    prev_ul = prev_h = prev_e = False

    for child in p_el:
        if child.tag != qn("w:r"):
            continue
        text = extract_text_from_run_xml(child)
        if not text:
            continue
        is_ul, is_hl, is_em = get_run_format_state_xml(child, sid2name, sid2based)

        closing = ""
        if prev_e and not is_em: closing += "</e>"
        if prev_h and not is_hl: closing += "</h>"
        if prev_ul and not is_ul: closing += "</ul>"

        opening = ""
        if not prev_ul and is_ul: opening += "<ul>"
        if not prev_h and is_hl: opening += "<h>"
        if not prev_e and is_em: opening += "<e>"

        tagged_parts.append(closing + opening + text)
        plain_parts.append(text)
        prev_ul, prev_h, prev_e = is_ul, is_hl, is_em

    tail = ""
    if prev_e: tail += "</e>"
    if prev_h: tail += "</h>"
    if prev_ul: tail += "</ul>"
    tagged_parts.append(tail)

    return "".join(plain_parts).strip(), "".join(tagged_parts).strip()


def split_card_text(plain_text: str, formatted_text: str):
    plain_paras = [p.strip() for p in plain_text.split("\n")]
    formatted_paras = [p.strip() for p in formatted_text.split("\n")]
    paired = [(p, f) for p, f in zip(plain_paras, formatted_paras) if p.strip()]
    if not paired:
        return "", "", "", ""
    cite_plain, cite_fmt = paired[0]
    body_plain = "\n".join(x[0] for x in paired[1:]).strip()
    body_fmt = "\n".join(x[1] for x in paired[1:]).strip()
    return cite_plain, cite_fmt, body_plain, body_fmt


def fast_parse_docx(path: str) -> List[Dict]:
    with zipfile.ZipFile(path, "r") as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Missing word/document.xml")
        doc_bytes = zf.read("word/document.xml")
        styles_bytes = zf.read("word/styles.xml") if "word/styles.xml" in zf.namelist() else b""

    sid2name, sid2based = parse_styles_xml(styles_bytes)
    doc_root = ET.fromstring(doc_bytes)
    body = doc_root.find("w:body", XML_NS)
    if body is None:
        return []

    cards = []
    inside_card = False
    current_heading = None
    plain_parts, tagged_parts = [], []

    def finalize():
        nonlocal current_heading, plain_parts, tagged_parts
        plain = "\n".join(x for x in plain_parts if x).strip()
        tagged = "\n".join(x for x in tagged_parts if x).strip()
        if plain:
            cp, cf, bp, bf = split_card_text(plain, tagged)
            cards.append({"tag": current_heading or "", "cite": cp, "plain_text": bp, "tagged_text": bf})
        current_heading, plain_parts, tagged_parts = None, [], []

    for child in body:
        if child.tag != qn("w:p"):
            continue
        sname = normalize_style_name(paragraph_style_name_xml(child, sid2name, sid2based))

        if sname in CARD_START_STYLE_NAMES:
            if inside_card:
                finalize()
            current_heading = paragraph_text_xml(child)
            plain_parts, tagged_parts = [], []
            inside_card = True
            continue
        if inside_card and sname in STOP_STYLE_NAMES:
            finalize()
            inside_card = False
            continue
        if inside_card:
            pp, tp = render_paragraph_from_xml(child, sid2name, sid2based)
            plain_parts.append(pp)
            tagged_parts.append(tp)

    if inside_card:
        finalize()
    return cards


# ══════════════════════════════════════════════════════════════════════
# DOCX RENDERING
# ══════════════════════════════════════════════════════════════════════

def render_docx_from_spans(tag: str, cite: str, text_body: str, spans, output_path: str):
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX

    starts, ends = {}, {}
    for s, e, t in spans:
        starts.setdefault(s, []).append(t)
        ends.setdefault(e, []).append(t)

    doc = Document()
    p = doc.add_paragraph()
    p.add_run(tag).bold = True
    p = doc.add_paragraph()
    p.add_run(cite)
    p = doc.add_paragraph()

    active, chunk = set(), []

    def flush():
        nonlocal chunk
        if not chunk:
            return
        run = p.add_run("".join(chunk))
        if "ul" in active: run.underline = True
        if "h" in active: run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if "e" in active: run.bold = True
        chunk = []

    for idx, ch in enumerate(text_body):
        if idx in ends:
            flush()
            for t in sorted(ends[idx], reverse=True):
                active.discard(t)
        if idx in starts:
            flush()
            for t in sorted(starts[idx]):
                active.add(t)
        chunk.append(ch)
    flush()
    doc.save(output_path)


# ══════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ══════════════════════════════════════════════════════════════════════

@mcp.tool()
def cut_card(
    tag: str,
    cite: str,
    text_body: str,
    top_k: int = 4,
) -> str:
    """
    Retrieve similar human-formatted debate cards for RAG-assisted formatting.

    Returns formatting instructions + retrieved examples so that Claude
    (in conversation) can format the new card directly. No API key needed.

    Args:
        tag: The card's tag line (argument claim).
        cite: The card's citation (author, date, source).
        text_body: The raw unformatted body text of the evidence.
        top_k: Number of similar examples to retrieve (default 4).

    Returns:
        Formatting instructions + retrieved examples + the new card to format.
        Claude should read these and return ONLY the tagged body text.
    """
    retrieved = retrieve_examples(tag, cite, text_body, top_k=top_k)

    example_blocks = []
    for i, ex in enumerate(retrieved, 1):
        example_blocks.append(
            f"--- Example {i} (similarity: {ex['score']:.4f}) ---\n"
            f"Example Tag: {ex['tag']}\n\n"
            f"Example Cite: {ex['cite']}\n\n"
            f"Example Input Body:\n{ex['plain_text'][:RETRIEVAL_TEXT_CHARS]}\n\n"
            f"Example Output Body:\n{ex['tagged_text'][:MAX_EXAMPLE_OUTPUT_CHARS]}"
        )

    examples_text = "\n\n".join(example_blocks)

    return (
        f"{FORMATTING_INSTRUCTIONS}\n\n"
        f"{'='*60}\n"
        f"RETRIEVED EXAMPLES FROM CORPUS\n"
        f"{'='*60}\n\n"
        f"Imitate the formatting style of these examples, not their wording.\n\n"
        f"{examples_text}\n\n"
        f"{'='*60}\n"
        f"NEW CARD TO FORMAT\n"
        f"{'='*60}\n\n"
        f"New Tag: {tag}\n"
        f"New Cite: {cite}\n"
        f"New Input Body:\n{text_body}\n\n"
        f"IMPORTANT: Return ONLY the tagged body text. "
        f"Preserve every word exactly. Only insert <ul>, <h>, <e> tags."
    )


@mcp.tool()
def search_cards(
    query_tag: str = "",
    query_cite: str = "",
    query_body: str = "",
    top_k: int = 5,
) -> str:
    """
    Search the FAISS index for similar debate cards.

    Args:
        query_tag: Tag/claim to search for.
        query_cite: Citation to search for.
        query_body: Body text to search for.
        top_k: Number of results (default 5).

    Returns:
        Matching cards with similarity scores and previews.
    """
    if not query_tag and not query_cite and not query_body:
        return "Error: Provide at least one of query_tag, query_cite, or query_body."

    retrieved = retrieve_examples(query_tag, query_cite, query_body, top_k=top_k)

    results = []
    for i, ex in enumerate(retrieved, 1):
        results.append(
            f"=== Result {i} (score: {ex['score']:.4f}) ===\n"
            f"Tag: {ex['tag'][:200]}\n"
            f"Cite: {ex['cite'][:200]}\n"
            f"Plain: {ex['plain_text'][:500]}\n"
            f"Tagged: {ex['tagged_text'][:500]}\n"
        )

    return "\n".join(results) if results else "No results found."


@mcp.tool()
def parse_docx(file_path: str) -> str:
    """
    Parse a Verbatim-formatted .docx and extract all debate cards.

    Args:
        file_path: Path to the .docx file.

    Returns:
        Extracted cards with tag, cite, body, and existing formatting.
    """
    path = os.path.expanduser(file_path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        cards = fast_parse_docx(path)
    except Exception as e:
        return f"Error parsing {path}: {e}"

    if not cards:
        return f"No cards found in {path}. Ensure the file uses Heading 4 style for card tags."

    output = f"Extracted {len(cards)} cards from {os.path.basename(path)}:\n\n"
    for i, card in enumerate(cards, 1):
        has_fmt = "<ul>" in card.get("tagged_text", "")
        output += (
            f"--- Card {i} ---\n"
            f"Tag: {card['tag'][:150]}\n"
            f"Cite: {card['cite'][:150]}\n"
            f"Body: {card['plain_text']}\n"
            f"Has formatting: {'Yes' if has_fmt else 'No'}\n\n"
        )

    return output


@mcp.tool()
def render_card_docx(
    tag: str,
    cite: str,
    tagged_text: str,
    output_path: str = "",
) -> str:
    """
    Render tagged text to a formatted .docx with underline/highlight/bold.

    Args:
        tag: The card's tag line.
        cite: The card's citation.
        tagged_text: Body with <ul>, <h>, <e> tags.
        output_path: Where to save (default: current directory).

    Returns:
        Path to the rendered .docx file.
    """
    plain_text, spans = tagged_text_to_spans(tagged_text)

    if not output_path:
        safe = re.sub(r"[^\w\s-]", "", tag[:50]).strip().replace(" ", "_") or "card"
        output_path = os.path.join(os.getcwd(), f"{safe}.docx")

    output_path = os.path.expanduser(output_path)
    render_docx_from_spans(tag, cite, plain_text, spans, output_path)
    return f"Card rendered to: {output_path}"


# ══════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
