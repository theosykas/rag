from .lexical_retrieval import LexicalSearch, SementicalSearch, HybridSearch
from .data_models import MinimalAnswer
from typing import List, Dict, Any
from json import JSONDecodeError
from pathlib import Path
import json
import uuid


class RagCli:
    def __init__(self, vllm_path: str) -> None:
        self.vllm_path = vllm_path
        self.lex_idx = "data/processed/bm25_index"
        self.lexical_engine = LexicalSearch(vllm_path=self.vllm_path,
                                            idx_path=self.lex_idx)
        self.sementical_idx = "data/processed/chromaDB_index"
        self.sementical_engine = SementicalSearch(vllm_path=self.vllm_path,
                                                  idx_path=self.sementical_idx)
        self.merge_search = HybridSearch(self.lexical_engine,
                                         self.sementical_engine)

    def index(self, max_chunk_size: int = 2000) -> str:
        self.lexical_engine.indexing(max_chunk_size)
        self.sementical_engine.indexing(max_chunk_size)
        return "Ingestion complete! Indices saved under data/processed/"

    def search(self, single_qwery: str) -> List[str]:
        pass

    # def search_dataset(self, search_path: str, k: int, saving_output:
    #                    str = "data/output/search_results_and_answer") -> None:
    #     try:
    #         with open(search_path, 'r', encoding='utf-8') as f:
    #             dataset = json.load(f)
    #         dataset = RagDataset(**dataset)
    #     except JSONDecodeError as e:
    #         print(f'{Fore.RED}[Error] {e}')
    #     return None

    def awnser(self, single_query: str, k: int = 10) -> str:
        # return awnser into json file
        hybrid_search = self.merge_search.relevant_search(single_query, k=k)
        qwen_awnser = "hello qwen"
        generate_awnser = MinimalAnswer(
            question_id=str(uuid.uuid4()),  # generate id
            question=single_query,
            retrieved_sources=hybrid_search,
            answer=qwen_awnser
        ).model_dump()
        try:
            output_path = Path("data/output/single_query")
            with open(output_path, "w", encoding="utf-8") as f:
                query = json.dump(generate_awnser, f, indent=4)
                return query
        except JSONDecodeError as e:
            print(f"[Error] {e}")
        return None

    # def awnser_dataset(self, generate_response: List[str]) -> List[RagRes]:
    #     pass

    def evaluate_res(self, data_ref: str) -> Dict[str, Any]:
        pass
