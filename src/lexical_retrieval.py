from .data_models import MinimalSource
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    Language,
)
from abc import ABC, abstractmethod
from typing import List, Any
from colorama import Fore
from pathlib import Path
import bm25s
import os


class BaseSearch(ABC):
    def __init__(self, vllm_path: str) -> None:
        self.vllm_path = vllm_path

    def chunk_vllm(self, max_chunk_size: int = 2000) -> List[dict]:
        chunks = []
        py_split = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON, chunk_size=max_chunk_size, chunk_overlap=20
        )
        md_split = MarkdownTextSplitter(chunk_size=max_chunk_size,
                                        chunk_overlap=20)
        default_split = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=20
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
            except (UnicodeDecodeError, OSError) as e:
                print(f"{Fore.RED}[Error] invalid file extension {e}")
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
    def indexing(self) -> List[Any]:
        pass

    @abstractmethod
    def search_engine(self) -> List[str]:
        pass


class LexicalSearch(BaseSearch):
    def __init__(self, vllm_path, idx_path: str):
        super().__init__(vllm_path)
        self.retriver = None
        self.idx_save = Path(idx_path)
        self.data_corpus = []

    def indexing(self) -> List[Any]:
        chunking_vllm = self.chunk_vllm(max_chunk_size=2000)
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

    def search_engine(self, qwery_user: str,
                      k: int = 10) -> List[MinimalSource]:
        qwery = bm25s.tokenize(qwery_user)
        scoring, res_k = self.retriver.retrieve(
            query_tokens=qwery, k=k)  # score/k
        return [MinimalSource(**item["source"]) for item in res_k[0]]
