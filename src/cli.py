from .retrieval import LexicalSearch, SementicalSearch, HybridSearch
from .data_models import (
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    MinimalSource,
    RagDataset,
    AnsweredQuestion
)
from typing import List, Dict, Any
from json import JSONDecodeError
from .qwen3_06B import Qwen
from pathlib import Path
from tqdm import tqdm
import uuid
import json
import os


class RagCli:
    def __init__(self, vllm_path: str) -> None:
        self.vllm_path = vllm_path
        self.lex_idx = "data/processed/bm25_index"
        self.lexical_engine = LexicalSearch(
            vllm_path=self.vllm_path, idx_path=self.lex_idx
        )
        self.sementical_idx = "data/processed/chromaDB_index"
        self.sementical_engine = SementicalSearch(
            vllm_path=self.vllm_path, idx_path=self.sementical_idx
        )
        self.merge_search = HybridSearch(self.lexical_engine,
                                         self.sementical_engine)
        self.qwen = None  # lazy load

    def load_qwen(self):
        if self.qwen is None:
            self.qwen = Qwen()
        return self.qwen

    def index(self, max_chunk_size: int = 2000) -> str:
        self.lexical_engine.indexing(max_chunk_size)
        self.sementical_engine.indexing(max_chunk_size)
        return "Ingestion complete! Indices saved under data/processed/"

    # rev search -> MinimalSRC
    def search(self, single_query: str, k: int = 10) -> List[str]:
        sources = self.merge_search.relevant_search(single_query, k=k)
        output_dir = Path("data/output/searchSingleQuery")
        output_path = output_dir / "search.json"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="UTF-8") as f:
                json.dump([source.model_dump() for source in sources],
                          f, indent=4)
                print(f"Sources search saved in {output_path}")
        except OSError as e:
            print(f"[Error] {e}")

    # compare rec@ll metrick
    def search_dataset(
        self,
        dataset_path: Path,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        try:
            with open(dataset_path, "r", encoding="UTF-8") as f:
                load_data = json.load(f)
            # minimal_search formt suject
            search_result = []
            for question in tqdm(load_data["rag_questions"],
                                 desc="search_dataset"):
                retrive_sources = self.merge_search.relevant_search(
                    question["question"], k=k
                )

                search_result.append(
                    MinimalSearchResults(
                        question_id=question["question_id"],
                        question=question["question"],  # in json dataset
                        retrieved_sources=retrive_sources,
                    ).model_dump()
                )

            output_search = StudentSearchResults(
                search_results=search_result, k=k
            ).model_dump()

            output_dir = Path(save_directory)
            os.makedirs(output_dir, exist_ok=True)
            output_file = output_dir / Path(dataset_path).name

            with open(output_file, "w", encoding="UTF-8") as f:
                student_dump = json.dump(output_search, f, indent=4)
                print(f"Saved student_search_results to {output_file}")
                return student_dump
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f"[Error] {e}")

    def awnser(self, single_query: str, k: int = 10) -> str:
        # return awnser into json file
        hybrid_search = self.merge_search.relevant_search(single_query, k=k)
        context = "\n\n".join([
            self.merge_search.get_text_chunk(src)
            for src in hybrid_search
        ])
        qwen_awnser = self.load_qwen().generate(query=single_query,
                                                context=context)

        generate_awnser = MinimalAnswer(
            question_id=str(uuid.uuid4()),  # generate id
            question=single_query,
            retrieved_sources=hybrid_search,
            answer=qwen_awnser,
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

    def answer_dataset(self, student_search_results_path: Path,
                       save_directory: Path):
        try:
            with open(student_search_results_path, "r", encoding="UTF-8") as f:
                data_load = json.load(f)
            awnser_stock = []
            for res in tqdm(data_load["search_results"],
                            desc="awnser generation"):
                context = "\n\n".join([
                    self.merge_search.get_text_chunk(MinimalSource(**src))
                    for src in res["retrieved_sources"]
                ])
                qwen_awnser = self.load_qwen().generate(query=res["question"],
                                                        context=context)

                awnser_stock.append(
                    MinimalAnswer(
                        question_id=res["question_id"],
                        question=res["question"],
                        retrieved_sources=res["retrieved_sources"],
                        answer=qwen_awnser
                    ).model_dump()
                )

            awnser_output = StudentSearchResultsAndAnswer(
                search_results=awnser_stock, k=10
            ).model_dump()

            output_dir = Path(save_directory)
            os.makedirs(output_dir, exist_ok=True)
            file = output_dir / Path(student_search_results_path).name

            with open(file, "w", encoding="UTF-8") as f:
                student_awnser = json.dump(awnser_output, f, indent=4)
                print(f"Saved student_search_results_and_answer to {file}")
                return student_awnser
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f"[Error] {e}")

    def get_overlap(self, retriverd_data: MinimalSource,
                    correct: MinimalSource) -> float:

        if retriverd_data.file_path != correct.file_path:
            return 0.0

        inter_1 = (retriverd_data.first_character_index,
                   retriverd_data.last_character_index)
        inter_2 = (correct.first_character_index,
                   correct.last_character_index)

        # min(0, 5)
        #     [0] [1]  pos tuple
        overlap = max(0, min(inter_1[1], inter_2[1]) -
                      max(inter_1[0], inter_2[0]))
        correct_data = inter_2[1] - inter_2[0]  # fin. - debut
        if correct_data == 0:
            return 0.0
        return overlap / correct_data

    def evaluate(self, student_answer_path: Path,
                 dataset_path: Path,
                 k: int,
                 max_context_length: int) -> Dict[str, Any]:
        try:
            with open(student_answer_path, 'r', encoding='UTF-8') as f:
                student_data = StudentSearchResults(**json.loads(f.read()))
            with open(dataset_path, 'r', encoding='UTF-8') as f:
                dataset = RagDataset(**json.loads(f.read()))
            recall_final = 0.0
            for i, quest in enumerate(dataset.rag_questions):
                if not isinstance(quest, AnsweredQuestion):
                    continue
                found = 0.0
                srcs_inter = quest.sources
                ret_srcs = student_data.search_results[i].retrieved_sources
                for src in srcs_inter:
                    for ret in ret_srcs:
                        if self.get_overlap(ret, src) >= 0.05:
                            found += 1
                            break
                if len(srcs_inter) > 0:
                    recall_final += found / len(srcs_inter)
                else:
                    recall_final == 0
            total_query = len(dataset.rag_questions)
            print(f'Recall@{student_data.k}:',
                  recall_final / total_query if total_query > 0 else 0)
        except Exception as e:
            print(f'[Error] {e}')


#   interval1:  |-------|          (de 0 à 7) retrive
#                   |-------|    (de 5 à 10) :correct

#              0    5    7   10
