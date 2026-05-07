from .data_models import MinimalSource
from abc import ABC, abstractmethod
from typing import List, Any
from pathlib import Path
import bm25s
import os


class BaseSearch(ABC):
    def __init__(self, vllm_path: str) -> None:
        self.vllm_path = vllm_path

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
                        ).model_dump(),  # serializable: json
                        "text": chunk_text
                        })
        except (UnicodeDecodeError, OSError) as e:
            print(f"[Error] {e}")
        # for c in chunks[:3]:
        #     print(c['source'])
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
            print(f'[Error] {e}')
        self.retriver.save(save_dir=str(self.idx_save),
                           corpus=chunking_vllm)

    def search_engine(self, qwery_user: str,
                      k: int = 10) -> List[MinimalSource]:
        qwery = bm25s.tokenize(qwery_user)
        scoring, res = self.retriver.retrieve(query_tokens=qwery, k=k)
        return [res['source'] for res in res[0]]
