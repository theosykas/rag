from .lexical_retrieval import LexicalSearch, SementicalSearch, HybridSearch
from .data_models import (
                          MinimalSearchResults,
                          StudentSearchResults,
                          MinimalAnswer)
from typing import List, Dict, Any
from json import JSONDecodeError
from pathlib import Path
import uuid
import json
import os


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

    def search(self, single_query: str, k: int = 10) -> List[str]:
        sources = self.merge_search.relevant_search(single_query, k=k)
        output_dir = Path("data/output")
        output_path = output_dir / "search.json"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='UTF-8') as f:
                json.dump([sources.model_dump() for sources in sources],
                          f, indent=4)
                print(f"Sources search saved in {output_path}")
        except OSError as e:
            print(f'[Error] {e}')

    # compare rec@ll metrick
    def search_dataset(self, dataset_path: Path,
                       k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        try:
            with open(dataset_path, "r", encoding='UTF-8') as f:
                load_data = json.load(f)
            # minimal_search formt suject
            search_result = []
            for question in load_data["rag_questions"]:
                retrive_sources = self.merge_search.relevant_search(
                    question["question"],
                    k=k)
                search_result.append(MinimalSearchResults(
                    question_id=question["question_id"],
                    question=question["question"],  # in json dataset
                    retrieved_sources=retrive_sources
                ).model_dump())

            output_search = StudentSearchResults(
                search_results=search_result,
                k=k
            ).model_dump()

            output_dir = Path(save_directory)
            os.makedirs(output_dir, exist_ok=True)
            output_file = output_dir / Path(dataset_path).name

            with open(output_file, "w", encoding='UTF-8') as f:
                student_dump = json.dump(output_search, f, indent=4)
                print(f"Saved student_search_results to {output_file}")
                return student_dump
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f'[Error] {e}')

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
            output_dir = Path("data/Output_SingleQuery")
            output_path = output_dir / "single_query.json"
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generate_awnser, f, indent=4)
                print(f"Saved student_signel_query to {output_path}")
        except OSError as e:
            print(f"[Error] {e}")

    # def awnser_dataset(self, generate_response: List[str]) -> List[RagRes]:
    #     pass

    def evaluate_res(self, data_ref: str) -> Dict[str, Any]:
        pass
