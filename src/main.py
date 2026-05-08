# from .retrieving_idx import get_embadded
# from .lexical_retrieval import LexicalSearch, SementicalSearch
# from .generator import Generator
from colorama import Fore
from .cli import RagCli
import fire


def main():
    # generate = Generator("dataset_docs_public.json")
    # model_emb = get_embadded("all-MiniLM-L6-v2")  # vector sementique search
    # lexical_retrive = LexicalSearch(vllm_path="vllm-0.10.1",
    #                                 idx_path="data/processed/bm25_index")
    # sementical_retrive = SementicalSearch(vllm_path="vllm-0.10.1",
    #                                       idx_path="data/processed/"
    #                                       "chromaDB_index")
    try:
        rag_cli = RagCli("vllm-0.10.1")
        fire.Fire(rag_cli)
        # generate.generate_json()
        # print(f"Indexing done in {elapsed:.1f}s / 300s max")
    except Exception as e:
        print(f'{Fore.RED}[ERROR] {e}')


if __name__ == "__main__":
    main()
