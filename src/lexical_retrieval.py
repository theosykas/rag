from .data_models import MinimalSource
from typing import List, Any
from pathlib import Path
import bm25s
from abc import ABC, abstractmethod


class BaseSearch(ABC):
    def __init__(self, vllm_path: str) -> None:
        self.vllm_path = vllm_path

    @abstractmethod
    def chunk_vllm(self, max_chunk_size: int = 2000) -> List[dict]:
        pass

    @abstractmethod
    def indexing(self) -> List[Any]:
        pass

    @abstractmethod
    def search_engine(self) -> List[str]:
        pass


class LexicalSearch(BaseSearch):
    def chunk_vllm(self, max_chunk_size: int = 2000) -> List[dict]:
        chunks = []
        try:
            for llm_file in Path(self.vllm_path).rglob("*"):
                if not llm_file.is_file():
                    continue
                data_analyse = llm_file.read_text(encoding="utf-8")
                for data_pos in range(0, len(data_analyse), max_chunk_size):
                    chunk_text = data_analyse[data_pos: data_pos +
                                              max_chunk_size]
                    chunks.append({
                        "source": MinimalSource(
                            file_path=str(llm_file),
                            first_character_index=data_pos,
                            last_character_index=min(data_pos + max_chunk_size,
                                                     len(data_analyse))
                        ),
                        "text": chunk_text
                        })
        except (UnicodeDecodeError, OSError) as e:
            print(e)
        # for c in chunks[:3]:
        #     print(c['source'])
        return chunks

    def indexing(self) -> List[Any]:
        chunking_vllm = self.chunk_vllm(max_chunk_size=2000)
        retriver_indexing = bm25s.BM25(corpus=chunking_vllm)
        retriver_indexing.index(bm25s.tokenize(chunking_vllm))

    def search_engine(self):
        pass
