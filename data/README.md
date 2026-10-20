# Public data package

`formatting_profile.json` is an aggregate summary of a local `cleaned_cards.json` containing 4,869 parsed debate cards. It includes body-length distributions, span counts, per-format coverage, and invalid-range counts. It contains no card text, taglines, citations, filenames, or per-card rows. The source JSON was produced by the preprocessing experiments in this repository; source-file lineage was not retained per card, so the exact origin of each record cannot be verified here.

`synthetic_card.json` shows the `tag`, `cite`, `text_body`, and formatting-range schema using invented text. Ranges use zero-based, end-exclusive Unicode code-point offsets. The example is a software fixture, not debate evidence.

To regenerate the profile from a compatible local JSON file (Node.js 18 or later):

```bash
node scripts/export_formatting_profile.mjs /path/to/cleaned_cards.json data/formatting_profile.json
```

## Why the full corpus is absent

The local collection includes OpenCaselist-derived disclosure files and full text from third-party evidence. [OpenCaselist's terms](https://opencaselist.com/terms) say uploaders retain their content and warn against copying or sharing others' protected content without the right to do so. Availability for disclosure is not itself a redistribution license.

There are two distinct published licenses worth separating:

| Material | Published license or permission | Relevance here |
| --- | --- | --- |
| [OpenCaselist application code](https://github.com/debate/debate-shared-research/blob/main/LICENSE) | GPL-3.0 | Covers that software, not every DOCX uploaded by debaters. No OpenCaselist code is copied into this repository. |
| [OpenDebateEvidence dataset](https://huggingface.co/datasets/Yusuf5/OpenCaselist) | Its curators label their processed dataset MIT and [report OpenCaselist's approval](https://papers.neurips.cc/paper_files/paper/2024/file/3c630d28df1cff44314d5798f82e02ec-Paper-Datasets_and_Benchmarks_Track.pdf). | A useful external dataset to consult under its own terms. This local JSON was independently parsed from disclosure files; we cannot assume that project's permission covers our separate copy or all underlying source publications. |

We did not find a blanket license or per-file permission covering a full-text release of this local collection. The full DOCX files, parsed card text, embeddings, search indexes, and pickled metadata are therefore excluded. A future corpus release would require documented rights for the included sources and a separate review for personal information.

The aggregate profile can be used to plan models and evaluate preprocessing quality. It cannot reproduce text-based experiments. The local input remains necessary for that work.
