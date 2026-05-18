from sentence_transformers import SentenceTransformer, CrossEncoder
from .data_models import MinimalSource
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
)
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from pathlib import Path
from tqdm import tqdm
import chromadb
import bm25s
import os


class BaseSearch(ABC):
    """. Abstract base class defining the blueprint for repository search
    engines.

    Provides core file chunking facilities tailored for specific file types
    and dictates abstract methods for downstream indexing and retrieval
    strategies.
    """
    def __init__(self, vllm_path: str, idx_path: str) -> None:
        """. Initializes common directory paths and tracking data models.

        Args:
            vllm_path (str): Target filesystem location of the raw codebase.
            idx_path (str): Destination directory where persistent
            index data is
                stored.
        """
        self.idx_save = Path(idx_path)
        self.vllm_path = vllm_path
        self.data_corpus: List[Dict[str, Any]] = []
        self.retriver: Any = None

    def chunk_vllm(self, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """. Parses, sections, and builds a chunked manifest of the codebase.

        Crawls target directories recursively and applies dedicated splitting
        heuristics (`PythonCodeTextSplitter`, `MarkdownTextSplitter`) to yield
        meaningful semantic bounds. Computes absolute character index
        locations.

        Args:
            max_chunk_size (int, optional): The absolute ceiling constraint for
                chunk lengths. Defaults to 2000.

        Returns:
            List[dict]: A collection of document fragment items paired with
                serialized tracking source metadata.
        """
        chunks = []
        py_split = PythonCodeTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200
        )
        md_split = MarkdownTextSplitter(chunk_size=max_chunk_size,
                                        chunk_overlap=200)
        default_split = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200  # ovelap === char
        )
        for llm_file in tqdm(list(Path(self.vllm_path).rglob("*")),
                             desc="Chunking_vllm"):
            if not llm_file.is_file():
                continue
            suffix = llm_file.suffix
            if suffix == ".py":
                splitter = py_split
            elif suffix == ".md":
                splitter = md_split
            else:
                splitter = default_split
            try:
                data_analyse = llm_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            chunk_text = splitter.split_text(data_analyse)
            search_idx_pos = 0
            for chunk in chunk_text:
                first_idx = data_analyse.find(chunk, search_idx_pos)
                if first_idx == -1:
                    first_idx = search_idx_pos  # error pos
                lst_idx = first_idx + len(chunk)
                chunks.append(
                    {
                        "source": MinimalSource(
                            file_path=str(llm_file),
                            first_character_index=first_idx,
                            last_character_index=lst_idx,
                        ).model_dump(),  # serializable: json
                        "text": chunk,
                    }
                )
                search_idx_pos = first_idx + 1  # check + 1
        return chunks

    @abstractmethod
    def indexing(self, max_chunk_size: int = 2000) -> None:
        """. Abstract interface designed to build and persist internal
        indices.

        Args:
            max_chunk_size (int, optional): Max size of a text block chunk.
                Defaults to 2000.
        """
        pass

    # find most pertinent chunk for response
    @abstractmethod
    def relevant_search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """. Abstract interface implemented to retrieve documents matching a
        query."""
        pass


class HybridSearch(BaseSearch):
    """. Orchestrator combining keyword matching and vector search with
    machine learning.

    Merges individual outputs from downstream semantic and lexical search
    layers,
    de-duplicates overlapping items, and uses a Neural Cross-Encoder to yield
    highly accurate reranked targets.
    """
    def __init__(self, lexical: 'LexicalSearch',
                 semtentical: 'SementicalSearch') -> None:
        """. Wireframe binding for the multi-engine hybrid pipeline
        infrastructure.

        Args:
            lexical (LexicalSearch): Instantiated BM25 keyword search module.
            semtentical (SementicalSearch): Instantiated ChromaDB vector
            engine.
        """
        self.model_ranking = CrossEncoder("ms-marco-MiniLM-L-6-v2")
        self.sementical_engine = semtentical
        self.lexical_engine = lexical

    def indexing(self, max_chunk_size: int = 2000) -> None:
        """. Placeholder overrider required by abstract schema validation."""
        raise NotImplementedError("pass def")

    # get all text for compare to rank
    def get_text_chunk(self, src: MinimalSource) -> str:
        """. Reconstructs and reads a localized slice of text string
        data from disk.

        Args:
            src (MinimalSource): Boundaries identifying file path and
                character indexing coordinates.

        Returns:
            str: The raw substring text isolated from the target resource file.
        """
        content = Path(src.file_path).read_text(encoding="utf-8")
        return content[src.first_character_index: src.last_character_index]

    def relevant_search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """. Executes coarse hybrid fetching followed by cross-attention
        re-scoring.

        Queries independent search algorithms, consolidates distinct candidate
        entries, eliminates overlapping elements, maps documents back to text,
        and scores query-context pairings using a Cross-Encoder to slice the
        top-k items.

        Args:
            query (str): The prompt parameter string submitted by the user.
            k (int, optional): The target size constraint bounding final
            returns.
                Defaults to 10.

        Returns:
            List[MinimalSource]: Sorted list of the most relevant source
                boundaries.
        """
        lex_res = self.lexical_engine.relevant_search(query, k=k)
        sem_res = self.sementical_engine.relevant_search(query, k=k)
        pairs_cross = []
        unique_source = self.check_duplicated(lex_res + sem_res)
        for raw_src in unique_source:
            chunk_txt = self.get_text_chunk(raw_src)
            pairs_cross.append([query, chunk_txt])
        scoring = self.model_ranking.predict(pairs_cross)
        sorted_score = sorted(zip(scoring, unique_source),
                              key=lambda x: x[0], reverse=True)
        return [src for score, src in sorted_score[:k]]  # final top K

    # check_duplicated chunking
    def check_duplicated(self,
                         src: List[MinimalSource]) -> List[MinimalSource]:
        """. Strips matching entry structures to ensure document
        collection uniqueness.

        Args:
            src (List[MinimalSource]): Collection containing potentially
            redundant
                source references.

        Returns:
            List[MinimalSource]: Cleaned collection containing only unique
                file segment bounds.
        """
        is_check = set()
        unique_data = []
        for chunk in src:
            identify_chunk = (chunk.file_path,
                              chunk.first_character_index,
                              chunk.last_character_index)
            if identify_chunk not in is_check:
                is_check.add(identify_chunk)
                unique_data.append(chunk)
        return unique_data


class LexicalSearch(BaseSearch):
    """. Keyword search engine implementing the BM25 statistical routing
    workflow."""
    def __init__(self, vllm_path: str, idx_path: str) -> None:
        """. Instantiates lexical boundaries and triggers immediate index
        retrieval."""
        super().__init__(vllm_path, idx_path)
        self.load_index()

    def indexing(self, max_chunk_size: int = 2000) -> None:
        """. Tokenizes document datasets and dumps BM25 index schemas
        to disk.

        Args:
            max_chunk_size (int, optional): Tokenizer window parameter length.
                Defaults to 2000.
        """
        chunking_vllm = self.chunk_vllm(max_chunk_size)
        self.data_corpus = chunking_vllm
        txt = [c["text"] for c in chunking_vllm]

        with tqdm(total=3, desc="bm25 indexing") as pbar:
            self.retriver = bm25s.BM25(corpus=chunking_vllm)
            token = bm25s.tokenize(texts=txt)
            pbar.update(1)
            self.retriver.index(token)
            pbar.update(1)

            try:
                os.makedirs(self.idx_save, exist_ok=True)
            except OSError as e:
                print(f"[Error] {e}")
            self.retriver.save(save_dir=str(self.idx_save),
                               corpus=chunking_vllm)
            pbar.update(1)

    def load_index(self) -> None:
        """. Reconstructs pre-computed lexical tables from disk
        if assets exist."""
        file_corpus = self.idx_save / "corpus.jsonl"  # "/" == path/theo
        params_file = self.idx_save / "params.index.json"
        if params_file.exists() and file_corpus.exists():
            self.retriver = bm25s.BM25.load(save_dir=str(self.idx_save),
                                            load_corpus=True)
            self.data_corpus = self.retriver.corpus

    def relevant_search(self, query_user: str,
                        k: int = 10) -> List[MinimalSource]:
        """. Performs exact keyword queries over the inverted BM25
        index structures.

        Args:
            query_user (str): Text containing keyword elements to extract.
            k (int, optional): Cutoff constraint bounding returned candidates.
                Defaults to 10.

        Returns:
            List[MinimalSource]: Best matching candidate items translated into
            schema targets.
        """
        qwery = bm25s.tokenize([query_user])
        res_k, scoring = self.retriver.retrieve(
                query_tokens=qwery, k=k)  # score/k
        # print(scoring)
        return [MinimalSource(**item["source"]) for item in res_k[0]]


class SementicalSearch(BaseSearch):
    """. Vector database search interface powered by ChromaDB and
    SentenceTransformers."""
    def __init__(self, vllm_path: str, idx_path: str) -> None:
        """22a. Establishes localized embeddings storage engines and links
        collections."""
        super().__init__(vllm_path, idx_path)
        self.client = chromadb.PersistentClient(str(self.idx_save))  # load_idx
        self.collection_chroma = self.client.get_or_create_collection("c_Vllm")

    def indexing(self, max_chunk_size: int = 2000,
                 batch_size: int = 5461) -> None:
        """. Generates neural text vector matrices and seeds local
        collections in batches.

        Processes the corpus text elements through an absolute embedding
        pipeline,
        slices datasets to guarantee transaction stability bounds, and saves
        vector
        keys along with metadata boundaries.

        Args:
            max_chunk_size (int, optional): Size ceiling bounding document
            parsing blocks.
                Defaults to 2000.
            batch_size (int, optional): Slicing index boundary to maintain
            transaction
                stability. Defaults to 5461.
        """
        chunking_vllm = self.chunk_vllm(max_chunk_size)
        self.data_corpus = chunking_vllm
        model_embedding = SentenceTransformer(
            "all-MiniLM-L6-v2")  # create emmbeding sent
        txt = [c["text"] for c in chunking_vllm]
        embeddings = model_embedding.encode(txt,
                                            show_progress_bar=True)
        for i in range(0, len(chunking_vllm), batch_size):
            end = min(i + batch_size, len(chunking_vllm))  # 0 i + 5000 12000
            self.collection_chroma.add(
                documents=txt[i:end],  # [0:5000]
                embeddings=embeddings[i:end],
                metadatas=[c["source"] for c in chunking_vllm[i:end]],
                ids=[str(data) for data in range(i, end)]  # idx all chunk
            )
        try:
            os.makedirs(self.idx_save, exist_ok=True)
        except OSError as e:
            print(f"[Error] {e}")

    def relevant_search(self, query_user: str,
                        k: int = 10) -> List[MinimalSource]:
        """. Performs top-k vector cosine similarity inquiries over the
        collection.

        Args:
            query_user (str): Natural language string describing user
            objective.
            k (int, optional): Size parameter constraint bounding vector
            responses.
                Defaults to 10.

        Returns:
            List[MinimalSource]: Collection of targeted source locations
            matching intent.
        """
        resultat = self.collection_chroma.query(
            query_texts=[query_user],
            n_results=k
        )
        return [MinimalSource(**item)
                for item in resultat["metadatas"][0]]
