from .retrieval import LexicalSearch, SementicalSearch, HybridSearch
from .data_models import (
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    MinimalSource,
    RagDataset,
    AnsweredQuestion,
)
from json import JSONDecodeError
from typing import Optional
from .qwen3_06B import Qwen
from pathlib import Path
from typing import List
from tqdm import tqdm
import uuid
import json
import os


class RagCli:
    def __init__(self, vllm_path: str) -> None:
        """RAG command-line helper.

        Args:
            vllm_path: Path to the local VLLM model or runtime used by
                retrieval and generation components.

        Initializes lexical (BM25) and semantical (ChromaDB) search
        engines and composes a hybrid search wrapper. Qwen model is
        lazy-loaded via load_qwen().
        """
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
        self.qwen: Optional[Qwen] = None

    def load_qwen(self) -> Qwen:
        """Lazy-load and return the Qwen model instance.

        Instantiates Qwen on first use and reuses the loaded instance
        for subsequent calls.
        """
        if self.qwen is None:
            self.qwen = Qwen()
        assert self.qwen is not None
        return self.qwen

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build or refresh the search indices.

        Args:
            max_chunk_size: Maximum number of characters per chunk used when
                processing documents for indexing.

        Returns:
            A status message confirming index ingestion completion.
        """
        self.lexical_engine.indexing(max_chunk_size)
        self.sementical_engine.indexing(max_chunk_size)
        return "Ingestion complete! Indices saved under data/processed/"

    def search(self, single_query: str, k: int = 10) -> List[MinimalSource]:
        """Run a hybrid retrieval for a single query and save results.

        Args:
            single_query: Query string to search over the indexed documents.
            k: Number of top-ranked sources to return.

        Returns:
            Retrieved source objects from the hybrid search.
        """
        sources = self.merge_search.relevant_search(single_query, k=k)
        output = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=single_query,
            retrieved_sources=sources
        ).model_dump()
        output_dir = Path("data/output/searchSingleQuery")
        output_path = output_dir / "search.json"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="UTF-8") as f:
                json.dump(output, f,
                          indent=4)
                print(f"Sources search saved in {output_path}")
        except OSError as e:
            print(f"[Error] {e}")
        return None

    def search_dataset(
        self,
        dataset_path: Path,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """. Executes a batch hybrid search over an entire evaluation dataset.

        Iterates through all questions within the provided JSON dataset file,
        queries the hybrid search engine (BM25 + ChromaDB + Cross-Encoder) for
        each item, formats the outputs into a validated schema using Pydantic,
        and serializes the results back to disk.

        Args:
            dataset_path (Path): Path to the input JSON dataset file containing
                the target 'rag_questions'.
            k (int, optional): The maximum number of relevant source chunks to
                retrieve for each query. Defaults to 10.
            save_directory (str, optional): The target directory path where the
                resulting JSON file will be stored. Defaults to
                "data/output/search_results".

        Returns:
            None: The function writes the serialized JSON output directly to
            disk.

        Raises:
            FileNotFoundError: Raised if the specified dataset file does not
            exist.
            JSONDecodeError: Raised if the source dataset file contains
            invalid JSON syntax.
        """
        try:
            with open(dataset_path, "r", encoding="UTF-8") as f:
                load_data = json.load(f)
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
                    )
                )

            output_search = StudentSearchResults(
                search_results=search_result, k=k
            ).model_dump()

            output_dir = Path(save_directory)
            os.makedirs(output_dir, exist_ok=True)
            output_file = output_dir / Path(dataset_path).name

            with open(output_file, "w", encoding="UTF-8") as f:
                json.dump(output_search, f, indent=4)
                print(f"Saved student_search_results to {output_file}")
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f"[Error] {e}")

    def answer(self, single_query: str, k: int = 10) -> None:
        """. Processes a single user query to retrieve context and generate a
        grounded answer.

        Executes a localized pipeline: retrieves the top-k most pertinent
        codebase
        chunks, reconstructs the code or documentation context sequence,
        passes the
        payload to the local Qwen LLM for inference, and records the complete
        transaction metadata into a single-query JSON log.

        Args:
            single_query (str): The natural language or technical question
                submitted by the user.
            k (int, optional): The number of document chunks to fetch and
            supply
                to the model's context window. Defaults to 10.

        Returns:
            str: The raw generated textual response returned by the local
                language model.

        Raises:
            OSError: Raised if the system encounters filesystem permission
            errors
                when generating the output tracking directory or file.
        """
        hybrid_search = self.merge_search.relevant_search(single_query, k=k)
        context = "\n\n".join(
            [self.merge_search.get_text_chunk(src) for src in hybrid_search]
        )
        qwen_awnser = self.load_qwen().generate(query=single_query,
                                                context=context)

        generate_awnser = MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=single_query,
            retrieved_sources=hybrid_search,
            answer=qwen_awnser,
        ).model_dump()

        try:
            output_dir = Path("data/output")
            output_path = output_dir / "single_query.json"
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generate_awnser, f, indent=4)
                print(f"Saved student_signel_query to {output_path}")
        except OSError as e:
            print(f"[Error] {e}")
        return None

    def answer_dataset(self, student_search_results_path: Path,
                       save_directory: Path) -> None:
        """. Generates LLM answers in batch using pre-computed search results.

        Loads a previously generated search result snapshot file, extracts the
        pre-retrieved source indices for each query, fetches the matching
        string
        contents from disk, prompts the local language model for answers, and
        consolidates the evaluation data into a unified output file.

        Args:
            student_search_results_path (Path): Path to the JSON file generated
                by the `search_dataset` step.
            save_directory (Path): Target directory path where the finalized
            answers
                and references package will be stored.

        Returns:
            None: The function writes the validated batch results directly to
            disk.

        Raises:
            FileNotFoundError: Raised if the search results index file cannot
            be found.
            JSONDecodeError: Raised if the input file contains malformed JSON
            data.
        """
        try:
            with open(student_search_results_path, "r", encoding="UTF-8") as f:
                data_load = json.load(f)
            awnser_stock = []
            total_len = len(data_load["search_results"])
            for i, res in enumerate(tqdm(data_load["search_results"],
                                         desc="awnser generation")):
                print(f"processed {i + 1} of {total_len} questions")
                context = "\n\n".join(
                    [
                        self.merge_search.get_text_chunk(MinimalSource(**src))
                        for src in res["retrieved_sources"]
                    ]
                )
                qwen_awnser = self.load_qwen().generate(
                    query=res["question"], context=context
                )

                awnser_stock.append(
                    MinimalAnswer(
                        question_id=res["question_id"],
                        question=res["question"],
                        retrieved_sources=res["retrieved_sources"],
                        answer=qwen_awnser,
                    )
                )

            awnser_output = StudentSearchResultsAndAnswer(
                search_results=awnser_stock, k=10
            ).model_dump()

            output_dir = Path(save_directory)
            os.makedirs(output_dir, exist_ok=True)
            file = output_dir / Path(student_search_results_path).name

            with open(file, "w", encoding="UTF-8") as f:
                json.dump(awnser_output, f, indent=4)
                print(f"Saved student_search_results_and_answer to {file}")
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f"[Error] {e}")

    def get_overlap(
        self, retriverd_data: MinimalSource, correct: MinimalSource
    ) -> float:
        """. Computes the character-level overlap ratio between a retrieved
        chunk and the ground truth.

        Evaluates the exact overlap alignment between two file sections.
        If the file paths
        do not match, the overlap ratio is instantly 0.0. Otherwise, it
        determines the
        intersection of character boundaries relative to the total length
        of the ground truth
        span.

        Args:
            retriverd_data (MinimalSource): The source chunk object generated
            by the retrieval pipeline.
            correct (MinimalSource): The golden standard / ground truth source
            chunk
                object from the dataset.

        Returns:
            float: The overlapping ratio coefficient, bounded between 0.0
            (no overlap)
                and 1.0 (complete coverage of the ground truth).
        """
        if retriverd_data.file_path != correct.file_path:
            return 0.0

        inter_1 = (
            retriverd_data.first_character_index,
            retriverd_data.last_character_index,
        )
        inter_2 = (correct.first_character_index, correct.last_character_index)

        # min(0, 5)
        #     [0] [1]Dict[str, Any]  pos tuple
        overlap = max(0, min(inter_1[1], inter_2[1]) -
                      max(inter_1[0], inter_2[0]))
        correct_data = inter_2[1] - inter_2[0]  # fin. - debut
        if correct_data == 0:
            return 0.0
        return overlap / correct_data

    def evaluate(
        self,
        student_answer_path: Path,
        dataset_path: Path,
        k: int,
        max_context_length: int,
    ) -> None:
        """. Computes the Recall@k performance metric for the search engine.

        Compares batch retrieval outputs against the evaluation dataset ground
        truths.
        A source chunk is successfully validated as "found" if its character-
        level text
        overlap meets or exceeds a strict 5% threshold ($0.05$). Calculates
        the overall
        mean recall across all processed questions.

        Args:
            student_answer_path (Path): Path to the generated student search
                results index file.
            dataset_path (Path): Path to the benchmark validation dataset
            containing
                the true target sources.
            k (int): The parameter constraint bounding the evaluation scope.
            max_context_length (int): The global context window limit applied
                during evaluations.

        Returns:
            Dict[str, Any]: A dictionary capturing execution performance
            statistics
                and metric aggregations.

        Raises:
            Exception: Generic catch-all handling block protecting the
            execution
                flow from corrupted dataset fields or indexing mismatches.
        """
        try:
            with open(student_answer_path, "r", encoding="UTF-8") as f:
                student_data = StudentSearchResults(**json.loads(f.read()))
            with open(dataset_path, "r", encoding="UTF-8") as f:
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
                    recall_final = 0
            total_query = len(dataset.rag_questions)
            print(
                f"Recall@{student_data.k}:",
                recall_final / total_query if total_query > 0 else 0,
            )
        except Exception as e:
            print(f"[Error] {e}")
