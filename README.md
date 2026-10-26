# Incrementally Maintained RWKV Memory as a Retrieval Guide for Small LLMs

Code and experiment notebooks for *Incrementally Maintained RWKV Memory as a Retrieval Guide for Small LLMs:
An Evaluation on LongMemEval*.

An RWKV-7 recurrent model reads a long chat history and hands clues or a draft answer to a small Llama reader.
The study asks whether that helps the reader recall facts from the history, and whether it matters if the
RWKV state is **rebuilt from scratch for every query** (re-ingest) or **maintained incrementally and persisted
between queries** (stream). Evaluation is on [LongMemEval_S](https://arxiv.org/abs/2410.10813).

> **All notebooks in this repository are written to run on Google Colab, and only on Colab.**
> They mount Google Drive, expect an NVIDIA GPU runtime (an A100-40GB is what they were designed and timed
> against), and stop with an explicit error if either is missing. There is no local or CPU code path and no
> `requirements.txt`: each notebook installs its own pinned dependencies in its second code section.

## Notebooks

Open a notebook in Colab with the badge, then follow [Running the experiments](#running-the-experiments).
Notebooks are numbered in the order the study builds on them.

| # | Notebook | What it does | Colab |
|---|---|---|---|
| 01 | [`01_memorag_rwkv7_2p9b_llama_1b_reader.ipynb`](notebooks/01_memorag_rwkv7_2p9b_llama_1b_reader.ipynb) | MemoRAG on LongMemEval_S with RWKV-7 G1 2.9B as the memory model and a Llama-3.2-1B reader | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShajeeAmjad/RWKV-Memory-Retrieval-For-Small-LLMs/blob/main/notebooks/01_memorag_rwkv7_2p9b_llama_1b_reader.ipynb) |
| 02 | [`02_memorag_rwkv_size_ladder_llama_8b_reader.ipynb`](notebooks/02_memorag_rwkv_size_ladder_llama_8b_reader.ipynb) | Adds a Llama-3.1-8B reader and an RWKV 1.5B / 2.9B / 7.2B memory-size ladder. Rebuilds the state from the whole history for every query: the **re-ingest condition** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShajeeAmjad/RWKV-Memory-Retrieval-For-Small-LLMs/blob/main/notebooks/02_memorag_rwkv_size_ladder_llama_8b_reader.ipynb) |
| 03 | [`03_incremental_vs_reingest_rwkv_state.ipynb`](notebooks/03_incremental_vs_reingest_rwkv_state.ipynb) | Master notebook: incremental (stream) vs. re-ingested state, decay analysis, and the revision experiments (negative control, equivalence test, all 500 questions, within-16k control) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShajeeAmjad/RWKV-Memory-Retrieval-For-Small-LLMs/blob/main/notebooks/03_incremental_vs_reingest_rwkv_state.ipynb) |
| 04 | [`04_incremental_vs_reingest_500q_run.ipynb`](notebooks/04_incremental_vs_reingest_500q_run.ipynb) | Same code as 03 with presets set for the 500-question run | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShajeeAmjad/RWKV-Memory-Retrieval-For-Small-LLMs/blob/main/notebooks/04_incremental_vs_reingest_500q_run.ipynb) |

### Provenance copies (do not re-run)

These are the records the paper's numbers were produced from. They are kept unmodified so the results can be
audited. To reproduce the study, run notebooks 01-04 instead.

| Notebook | What it is |
|---|---|
| [`notebooks/provenance/03_incremental_vs_reingest_FROZEN_pre_revision_2026-10-23.ipynb`](notebooks/provenance/03_incremental_vs_reingest_FROZEN_pre_revision_2026-10-23.ipynb) | Notebook 03 as it was when the main stream vs. re-ingest runs printed their results (first 94 cells of 03, before the revision sections were appended) |
| [`notebooks/provenance/04_incremental_vs_reingest_500q_EXECUTED_2026-10-25.ipynb`](notebooks/provenance/04_incremental_vs_reingest_500q_EXECUTED_2026-10-25.ipynb) | The executed 500-question run, outputs included |

### Earlier file names

The notebooks were renamed for this release. Prose and code comments inside the notebooks still use the
original names, which map as follows.

| Original name | Current name |
|---|---|
| `longmemeval_memorag.ipynb`, `longmemeval_memorag_final.ipynb` | `01_memorag_rwkv7_2p9b_llama_1b_reader.ipynb` |
| `longmemeval_memorag_ext_new.ipynb` | `02_memorag_rwkv_size_ladder_llama_8b_reader.ipynb` |
| `longmemeval_stream_state.ipynb` | `03_incremental_vs_reingest_rwkv_state.ipynb` |
| `longmemeval_stream_state_run500.ipynb` | `04_incremental_vs_reingest_500q_run.ipynb` |
| `longmemeval_stream_state.pre-revision-2026-10-23.ipynb` | `provenance/03_incremental_vs_reingest_FROZEN_pre_revision_2026-10-23.ipynb` |
| `longmemeval_stream_state_run500.executed-2026-10-25.ipynb` | `provenance/04_incremental_vs_reingest_500q_EXECUTED_2026-10-25.ipynb` |

## Requirements

| Requirement | Detail |
|---|---|
| Platform | Google Colab with a GPU runtime (`Runtime > Change runtime type`). A100-40GB recommended; notebooks print a warning below 38 GB and refuse to start with no GPU |
| Storage | Google Drive, mounted at `/content/drive`. All caches, generations and judgements are written to `MyDrive/lme_memorag/` so a dropped runtime loses at most one in-flight chunk |
| `OPENAI_API_KEY` | Required by all notebooks. Used for the judge (`gpt-4o-2024-08-06`) |
| `HF_TOKEN` | Required by all notebooks. Hugging Face token for the gated Llama weights; accept the Llama-3.2 and Llama-3.1 licences on Hugging Face first |
| `OPENROUTER_API_KEY` | Required by notebooks 02, 03 and 04 (Llama-3.1-8B reader, served through OpenRouter) |
| Python packages | Installed by each notebook's install section: `vllm==0.28.0`, `torch==2.13.0+cu132`, `torchvision==0.28.0+cu132`, `rwkv`, `sentence-transformers`, `openai>=1.40`, `tokenizers>=0.20`, plus `numpy`, `pandas`, `matplotlib`, `tqdm` |

Provide the keys either as Colab Secrets (the key icon in the Colab sidebar) or as `KEY=value` lines in a
`.env` file at `MyDrive/lme_memorag/.env`. Never paste keys into a notebook cell.

## Running the experiments

1. Open the notebook in Colab (badge above), select a GPU runtime, and add the keys.
2. Run from the top with `Runtime > Run all`. The first code section checks for Drive and a GPU, then the
   install section pins the dependencies. If the install section says `torch` is already imported, use
   `Runtime > Restart session` and run again from the top.
3. Run the notebooks in numerical order. Notebook 02 reuses cached assets from 01 where present, and
   notebook 03 reads notebook 02's results from the shared Drive folder (the re-ingest numbers and the
   question subset), so **02 must have completed before 03**. Notebook 04 needs the same.
4. Each notebook has a scale preset near the top. Set the preset before running; the larger presets cost
   real GPU time and API spend. Notebook 02 targets under six hours on an A100-40GB.

Re-runs are resumable: finished work is cached on Drive and skipped.

## Data and models

Nothing needs to be downloaded by hand; the notebooks fetch everything into Drive on first run.

- **Dataset:** [`longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
  using the LongMemEval_S split ([Wu et al., ICLR 2025](https://arxiv.org/abs/2410.10813)).
- **Memory models:** RWKV-7 G1 (1.5B, 2.9B, 7.2B) checkpoints from
  [`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1), loaded with the inference-only `rwkv` package.
- **Readers:** `meta-llama/Llama-3.2-1B-Instruct` (local, vLLM) and `meta-llama/Llama-3.1-8B-Instruct`.
- **Retriever:** [`NovaSearch/stella_en_1.5B_v5`](https://huggingface.co/NovaSearch/stella_en_1.5B_v5).
- **Judge:** `gpt-4o-2024-08-06`, using LongMemEval's question-type judging templates.

## Repository layout

```
notebooks/
  01_memorag_rwkv7_2p9b_llama_1b_reader.ipynb
  02_memorag_rwkv_size_ladder_llama_8b_reader.ipynb
  03_incremental_vs_reingest_rwkv_state.ipynb
  04_incremental_vs_reingest_500q_run.ipynb
  provenance/                 frozen and executed copies the paper's numbers come from
```

## Notes on reproducibility

- LLM-judged accuracy depends on API-served models (OpenAI judge, OpenRouter-hosted 8B reader), so small
  run-to-run differences are expected even at temperature 0. Notebook 03 measures the corresponding noise
  floor directly (its gates G6-G10) and reports it with every comparison.
- Stream and re-ingest states are mathematically identical for RWKV-7; notebook 03 treats their agreement as
  a correctness check, not as a finding.

## Citation

```bibtex
@misc{rwkv_memory_longmemeval,
  title  = {Incrementally Maintained {RWKV} Memory as a Retrieval Guide for Small {LLM}s:
            An Evaluation on {LongMemEval}},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/ShajeeAmjad/RWKV-Memory-Retrieval-For-Small-LLMs}
}
```
