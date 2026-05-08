from sentence_transformers import SentenceTransformer, CrossEncoder
from .data_models import MinimalSource
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
)
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from colorama import Fore, Style
from pathlib import Path
import chromadb
import bm25s
import os


class BaseSearch(ABC):
    def __init__(self, vllm_path: str, idx_path: str) -> None:
        self.idx_save = Path(idx_path)
        self.vllm_path = vllm_path
        self.data_corpus = []
        self.retriver = None

    def chunk_vllm(self, max_chunk_size: int = 2000) -> List[dict]:
        chunks = []
        py_split = PythonCodeTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200
        )
        md_split = MarkdownTextSplitter(chunk_size=max_chunk_size,
                                        chunk_overlap=200)
        default_split = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200  # ovelap === char
        )
        for llm_file in Path(self.vllm_path).rglob("*"):
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
                print(f"invalid file extension {Fore.GREEN}{suffix}{Style.RESET_ALL}")
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
    def indexing(self, max_chunk_size: int = 2000) -> List[Any]:
        pass

    # find most pertinent chunk for response
    @abstractmethod
    def relevant_search(self) -> List[str]:
        pass


class HybridSearch(BaseSearch):
    def __init__(self, lexical: LexicalSearch, semtentical: SementicalSearch) -> None:
        self.lexical_engine = lexical
        self.sementical_engine = semtentical
        self.model_ranking = CrossEncoder("ms-marco-MiniLM-L-6-v2")

    def relevant_search(self, query: str, k: int = 10):
        lex_res = self.lexical_engine.relevant_search([query], k=k)
        sem_res = self.sementical_engine.relevant_search([query], k=k)
        pairs_cross = []
        unique_source = self.check_duplicated(lex_res + sem_res)
        for res in unique_source:
            chunk_txt =
            pairs_cross.append([query, chunk_txt])
        scoring = self.model_ranking.predict(pairs_cross)
        return unique_source[:k]  # final top K

    def check_duplicated(src: List[MinimalSource]) -> List[MinimalSource]:
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
    def __init__(self, vllm_path, idx_path: str):
        super().__init__(vllm_path, idx_path)

    def indexing(self, max_chunk_size: int = 2000) -> List[Any]:
        chunking_vllm = self.chunk_vllm(max_chunk_size)
        # print(f"lexical idx = {max_chunk_size}")
        self.data_corpus = chunking_vllm
        txt = [c["text"] for c in chunking_vllm]

        self.retriver = bm25s.BM25(corpus=chunking_vllm)
        token = bm25s.tokenize(texts=txt)
        self.retriver.index(token)

        try:
            os.makedirs(self.idx_save, exist_ok=True)
        except OSError as e:
            print(f"[Error] {e}")
        self.retriver.save(save_dir=str(self.idx_save), corpus=chunking_vllm)

    def relevant_search(self, query_user: List[str],
                        k: int = 10) -> List[MinimalSource]:
        qwery = bm25s.tokenize(query_user)
        scoring, res_k = self.retriver.retrieve(
                query_tokens=qwery, k=k)  # score/k
        return [MinimalSource(**item["source"]) for item in res_k[0]]


class SementicalSearch(BaseSearch):
    def __init__(self, vllm_path: str, idx_path: str) -> None:
        super().__init__(vllm_path, idx_path)
        self.client = chromadb.PersistentClient(str(self.idx_save))
        self.collection_chroma = self.client.get_or_create_collection("c_Vllm")

    def indexing(self, max_chunk_size: int = 2000,
                 batch_size: int = 5461) -> List[Dict]:
        chunking_vllm = self.chunk_vllm(max_chunk_size)
        self.data_corpus = chunking_vllm
        model_embedding = SentenceTransformer(
            "all-MiniLM-L6-v2")  # create emmbeding sent
        txt = [c["text"] for c in chunking_vllm]
        embeddings = model_embedding.encode(txt,
                                            show_progress_bar=True)
        for i in range(0, len(chunking_vllm), batch_size):
            end = min(i + batch_size, len(chunking_vllm))  # 0 iter + 5000 12000
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

    def relevant_search(self, query_user: List[str],
                        k: int = 10) -> List[MinimalSource]:
        resultat = self.collection_chroma.query(
            query_texts=query_user,
            n_results=k
        )
        return [MinimalSource(**item)
                for item in resultat["metadatas"][0]]
