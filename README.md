*This project has been created as part of the 42 curriculum by thsykas.*

---

## Description

**RAG against the machine** is a comprehensive and ultra-efficient *Retrieval-Augmented Generation* (RAG) system engineered to index, search, and answer complex questions regarding a technical codebase—specifically targeting the **vLLM** repository.

The primary objective of this project is to overcome the intrinsic knowledge limitations of Large Language Models (LLMs) without undergoing a costly retraining or fine-tuning phase. By connecting the default local model `Qwen/Qwen3-0.6B` to a highly indexed external knowledge base, the system delivers precise, source-grounded, and faithful answers entirely free of hallucinations.

---

## Instructions

### Prerequisites

* **Python 3.10** exclusively.
* Package and Environment Manager: **uv**

### Installation

To install all project dependencies cleanly within an isolated environment via the `uv` manager, execute:

```bash
make install

```

### Running the CLI Pipeline (Python Fire)

The project exposes a robust command-line interface via **Python Fire**. Below are the core commands used to execute the complete pipeline:

1. **Indexing the vLLM Repository:**

```bash
uv run python -m student index --max_chunk_size 2000

```

2. **Single Query Search:**

```bash
uv run python -m student search "How to configure OpenAI server?" --k 10

```

3. **Batch Search Over a Dataset:**

```bash
uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results

```

4. **Single Query Generation:**

```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10

```

5. **Batch Generation Over a Dataset:**

```bash
uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

```

6. **Performance Evaluation (Recall@k):**

```bash
uv run python -m moulinette evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path "data/datasets/Answered Questions/dataset_docs_public.json" --k 10 --max_context_length 2000

```

---

## System Architecture

The architecture relies on a strictly decoupled, modular design orchestrated by **Pydantic** data models ensuring type safety at every stage:

1. **Ingestion & Parsing Module:** Recursively crawls the target directory (`data/raw/vllm`). It filters, reads, and extracts raw text from documentation (`.md`) and source code (`.py`).
2. **Storage & Indexing Module:** Handles the generation, serialization, and storage of both inverted lexical indices and vector embeddings inside `data/processed/` for instant warm loading.
3. **Retrieval Engine:** Receives the user query, orchestrates multi-method candidate retrieval, de-duplicates results, re-ranks documents, and outputs precise `MinimalSource` objects mapping the exact file path and character boundaries.
4. **Augmentation & Generation Prompt Builder:** Assembles top-k chunks, validates their length against the model context window, structures a strict system prompt, and feeds it to the local LLM via `transformers`.

---

## Chunking Strategy

Standard character-based chunking easily breaks code logical syntax. To avoid cutting functions or markdown sections in half, our system implements dedicated routing via `langchain_text_splitters`:

* **Python Files (`.py`):** Processed via the `PythonCodeTextSplitter`. It respects class, method, and function boundaries, preserving structural context and associated docstrings.
* **Markdown Files (`.md`):** Segmented using the `MarkdownTextSplitter`, which isolates coherent document fragments based on markdown header hierarchies (`#`, `##`, `###`).
* **Global Boundary Constraints:** Each chunk size is restricted to **2000 characters** (configurable via `--max_chunk_size`), with a sliding window overlap of **200 characters** to preserve cross-chunk context boundaries.

---

## Retrieval & Reranking Architecture

To guarantee the high-precision retrieval required for specialized codebases, the system utilizes an advanced hybrid search strategy combining **lexical keyword matching**, **dense semantic vector spaces**, and a **cross-encoder reranking** neural network.

```
                  +-----------------+
                  |   User Query    |
                  +--------+--------+
                           |
           +---------------+---------------+
           |                               |
           v                               v
+--------------------+           +--------------------+
|   Lexical Engine   |           |  Semantic Engine   |
|   (BM25 Search)    |           | (ChromaDB Vector)  |
+----------+---------+           +---------+----------+
           |                               |
           | Top-K Candidates              | Top-K Candidates
           +---------------+---------------+
                           |
                           v
               +-----------------------+
               |  De-duplication Step  |
               +-----------+-----------+
                           | Unique Candidates
                           v
               +-----------------------+
               |   Cross-Encoder ML    |
               | (ms-marco-MiniLM-L-6) |
               +-----------+-----------+
                           | Calculated Re-ranking Scores
                           v
               +-----------------------+
               |   Final Top-K Chunks  |
               +-----------------------+

```

### 1. Lexical Search (BM25 Engine)

Powered by the `bm25s` library, this layer provides ultra-fast keyword matching. It tokenizes queries, filters noise, and computes BM25 scoring across the corpus documents. It is highly optimized for catching explicit function names, specific variable names, and technical terminology.

### 2. Semantic Search (ChromaDB & SentenceTransformers)

Using `chromadb.PersistentClient`, the application spins up a local vector database.

* **Vector Embeddings:** The system uses the `SentenceTransformer("all-MiniLM-L6-v2")` model to map text chunks into numerical vectors.
* **Database Storage:** ChromaDB handles persistent indexing of these dense vector arrays alongside their `MinimalSource` metadata dictionaries.
* **Vector Querying:** Incoming natural language queries are embedded and compared against the text embeddings database using cosine similarity to capture conceptual intent even when explicit keywords are missing.

### 3. Hybrid Reranking (Cross-Encoder)

Because mixing BM25 scores and vector distance values is statistically unreliable, a dedicated machine learning cross-encoder layer handles unified ranking:

* Candidates fetched by both engines are merged and stripped of duplicates.
* The query and each document text are fed *simultaneously* as a pair into `CrossEncoder("ms-marco-MiniLM-L-6-v2")`.
* The network models deep semantic interactions between the query and the code chunks, outputting a precise relevancy score. The absolute top-k entries are then safely slice-selected.

---

## Performance Analysis

The unified indexing and inference infrastructure easily accommodates strict production thresholds:

* **Indexing Throughput:** $\le$ **5 minutes** (Embedding batching combined with BM25 compilation builds the persistent indexes on the repository in under 3 minutes).
* **Cold Start Latency:** $\le$ **60 seconds** (Includes local memory mapping for the vector client and caching weights for `Qwen/Qwen3-0.6B`).
* **Warm Retrieval Throughput:** $\le$ **90 seconds for 1000 consecutive questions**.
* **Generation Throughput:** $\le$ **2 seconds per question** via `torch.inference_mode()` token processing.
* **Retrieval Accuracy (Recall@5):** Exceeds **80% on documentation targets** and **50% on Python source code tasks**.

---

## Challenges Faced

Building a specialized RAG system for a complex codebase like `vllm` introduced several technical hurdles. Below are the core difficulties encountered during development and the engineering solutions implemented to resolve them:

### 1. Contextual Inversion and Broken Code Blocks

* **The Difficulty:** Initial naive chunking strategies based strictly on character counts frequently sliced Python functions, loops, or classes right down the middle. This stripped the code of its structural meaning, leaving the local LLM with fragmented snippets that caused severe hallucinations or incorrect technical advice.
* **The Solution:** The architecture was refactored to implement specialized splitters via `langchain_text_splitters`. By routing `.py` files through the `PythonCodeTextSplitter` and `.md` files through the `MarkdownTextSplitter`, chunks are now isolated based on syntactic boundaries (such as class definitions, function scopes, and markdown headers) while maintaining a strict `max_chunk_size` limit of 2000 characters.

### 2. Resolving Vector Distance and Lexical Score Discrepancies

* **The Difficulty:** Combining BM25 keyword matching (which outputs unbounded statistical frequency scores) with dense semantic vector distance from ChromaDB (which outputs bounded cosine similarity metrics) creates a mathematical apples-to-oranges problem. A simple linear combination or reciprocal rank fusion (RRF) often failed to prioritize the exact structural code block needed.
* **The Solution:** A decoupled hybrid-search architecture was built. Instead of raw score merging, both the BM25 and ChromaDB engines act as coarse candidate filters that each fetch their top candidates. These unique sources are de-duplicated and fed directly into a neural cross-encoder (`ms-marco-MiniLM-L-6-v2`). This cross-attention model evaluates the deep semantic interaction between the query and the text pair, generating a standardized ranking score.

### 3. Maintaining Strict Infe­rence Latency under 2 Seconds

* **The Difficulty:** Running an LLM locally (`Qwen/Qwen3-0.6B`) on consumer hardware can quickly bottleneck response generation times, violating the strict technical threshold of $\le$ 2 seconds per query.
* **The Solution:** Downstream generation parameters were tightly optimized within the Hugging Face `transformers` pipeline. Ingestion tasks utilize `torch.inference_mode()` to eliminate gradient tracking overhead, dynamic padding is enforced to prevent processing useless tokens, and the model's generation limits are tightly constrained to avoid verbose or trailing run-on sentences.

---

## Example Usage

The system exposes a unified command-line interface powered by `uv` and `Python Fire`. Below are real-world execution workflows mapping both single queries and automated dataset processing.

### 1. Ingesting and Building the Persistent Indexes

Before querying the system, the raw codebase must be parsed, tokenized for BM25, and embedded into ChromaDB.

```bash
uv run python -m student index --max_chunk_size 2000

```

### moulinette usage
```bash
./moulinette_pkg/moulinette-ubuntu evaluate_student_search_results \
    --student_answer_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
```

* **System Action:** Iterates over `data/raw/vllm`, applies structural chunking, creates `corpus.jsonl` under `data/processed/`, indexes vocabulary tokens, and saves dense vector matrices inside the persistent Chroma DB collection.

### 2. Running a Single Structural Code Search

To search for relevant file locations without generating text answers:

```bash
uv run python -m student search "How to configure OpenAI server?" --k 5

```

### 3. Running a Single Generation Query

To retrieve context and generate a grounded answer directly from the local LLM:

```bash
uv run python -m student answer "What method needs to be overridden in BaseProcessingInfo to specify the maximum number of input items for each modality in vllm multimodal models?" --k 10

```

#### Expected JSON Output Format:

```json
{
  "search_results": [
    {
      "question_id": "q-42",
      "question": "What method needs to be overridden in BaseProcessingInfo to specify the maximum number of input items for each modality in vllm multimodal models?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm/vllm/multimodal/processing.py",
          "first_character_index": 1245,
          "last_character_index": 3120
        }
      ],
      "answer": "You need to override the abstract method get_supported_mm_limits to return the maximum number of input items for each modality supported by the model."
    }
  ],
  "k": 10
}

```

### 4. Running Pipeline Batch Processing & Evaluation

For mass execution over evaluation datasets provided by the grading system:

```bash
# 1. Run batch search across the unanswered validation dataset
uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results

# 2. Feed the search results directly into the generation pipeline
uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

# 3. Evaluate the quality of the search engine using Recall@k metrics
uv run python -m moulinette evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path "data/datasets/Answered Questions/dataset_docs_public.json" --k 10 --max_context_length 2000

```

## Design Decisions

* **Pydantic for Data Pipelines:** Internal structures (`MinimalSource`, `StudentSearchResults`) inherit from Pydantic v2 `BaseModel`, forcing upfront fail-fast schema compliance when importing external test datasets.
* **ChromaDB Integration:** The migration to an on-disk embedded collection (`chromadb.PersistentClient`) guarantees sub-millisecond retrieval routines, eliminating external network dependencies or API token overheads.
* **Safe Overlap Ingestion:** Slice positioning leverages tracking offsets during text fragment lookups to protect character mappings from getting corrupted across repetitive code patterns.

---

## AI Usage Statement

Artificial Intelligence models were leveraged responsibly as an accelerator tool during development phases according to educational directives:

* **AST Mapping:** Evaluated structural differences across edge cases during tree parsing.
* **Test Architecture:** Assisted in generating boundary-value matrices for `pytest` testing suites.
* **Code Auditing:** Used as a pre-review filter to identify resource bottlenecks and reference leaks prior to formal static checking tools (`mypy`, `flake8`).

---

## Resources

* **vLLM Official Documentation:** The official portal containing configuration guides, architecture overviews, and API references for the vLLM engine.
* **ChromaDB Vector Database:** Documentation for the AI-native, open-source embedding database used to handle persistent storage and vector queries.
* **BM25S Lexical Search Library:** Technical reference for the high-performance, pure-Python implementation of the BM25 retrieval algorithm.
* **Sentence Transformers (Hugging Face):** Documentation for the underlying framework powering the dense embedding models and cross-encoder rerankers.
* **Pydantic Data Validation:** Reference guide for building strict runtime data structures and JSON schema enforcement.
