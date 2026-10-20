#!/usr/bin/env node

// Export aggregate formatting measurements without copying card text or metadata.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('Usage: node scripts/export_formatting_profile.mjs INPUT_JSON OUTPUT_JSON');
  process.exit(2);
}
if (resolve(inputPath) === resolve(outputPath)) {
  throw new Error('Input and output paths must differ');
}

const cards = JSON.parse(readFileSync(inputPath, 'utf8'));
if (!Array.isArray(cards)) throw new Error('Input must be a JSON array of cards');

const kinds = ['underline_ranges', 'emphasis_ranges', 'highlight_ranges'];
const buckets = [0, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000];
const histogram = Object.fromEntries(
  buckets.map((low, i) => [
    i + 1 < buckets.length ? `${low}-${buckets[i + 1] - 1}` : `${low}+`,
    0,
  ]),
);
const byKind = Object.fromEntries(kinds.map(kind => [kind, {
  cards_with_spans: 0,
  span_count: 0,
  valid_span_count: 0,
  invalid_span_count: 0,
  valid_span_characters: 0,
  covered_characters_within_kind: 0,
}]));
const lengths = [];
const spansPerCard = [];
let nonemptyBodies = 0;
let cardsWithInvalidSpans = 0;

function quantiles(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = p => sorted[Math.floor((sorted.length - 1) * p)];
  return { min: sorted[0], median: at(0.5), p90: at(0.9), p99: at(0.99), max: sorted.at(-1) };
}

function unionLength(ranges) {
  if (!ranges.length) return 0;
  ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let covered = 0;
  let [start, end] = ranges[0];
  for (const [nextStart, nextEnd] of ranges.slice(1)) {
    if (nextStart > end) {
      covered += end - start;
      [start, end] = [nextStart, nextEnd];
    } else {
      end = Math.max(end, nextEnd);
    }
  }
  return covered + end - start;
}

for (const card of cards) {
  if (!card || typeof card.text_body !== 'string') {
    throw new Error('Each card must contain a text_body string');
  }
  // The source parser is Python; its offsets count Unicode code points.
  const bodyLength = Array.from(card.text_body).length;
  lengths.push(bodyLength);
  if (bodyLength > 0) nonemptyBodies++;
  const key = buckets.findLast(low => bodyLength >= low);
  const bucket = buckets.indexOf(key);
  histogram[bucket + 1 < buckets.length ? `${key}-${buckets[bucket + 1] - 1}` : `${key}+`]++;

  let cardSpanCount = 0;
  let hasInvalidSpan = false;
  for (const kind of kinds) {
    const ranges = card[kind];
    if (!Array.isArray(ranges)) throw new Error(`${kind} must be an array`);
    const stats = byKind[kind];
    if (ranges.length) stats.cards_with_spans++;
    cardSpanCount += ranges.length;
    stats.span_count += ranges.length;
    const valid = [];
    for (const range of ranges) {
      if (!Array.isArray(range) || range.length !== 2 ||
          !Number.isInteger(range[0]) || !Number.isInteger(range[1]) ||
          range[0] < 0 || range[1] <= range[0] || range[1] > bodyLength) {
        stats.invalid_span_count++;
        hasInvalidSpan = true;
        continue;
      }
      stats.valid_span_count++;
      stats.valid_span_characters += range[1] - range[0];
      valid.push(range);
    }
    stats.covered_characters_within_kind += unionLength(valid);
  }
  spansPerCard.push(cardSpanCount);
  if (hasInvalidSpan) cardsWithInvalidSpans++;
}

const profile = {
  schema_version: 1,
  description: 'Aggregate, text-free measurements of a local CardCutter preprocessing output.',
  cards: cards.length,
  nonempty_bodies: nonemptyBodies,
  body_length_codepoints: quantiles(lengths),
  body_length_histogram_codepoints: histogram,
  spans_per_card: quantiles(spansPerCard),
  cards_with_invalid_spans: cardsWithInvalidSpans,
  formatting: byKind,
  notes: [
    'Ranges are zero-based, end-exclusive Unicode code-point offsets in text_body.',
    'A valid range has integer endpoints with 0 <= start < end <= body length.',
    'Valid span characters count overlaps; covered characters are the union within each formatting kind.',
    'The profile does not contain tags, citations, body text, filenames, or per-card records.',
  ],
};

writeFileSync(outputPath, `${JSON.stringify(profile, null, 2)}\n`, 'utf8');
console.log(`Wrote text-free profile for ${cards.length} cards to ${outputPath}`);
