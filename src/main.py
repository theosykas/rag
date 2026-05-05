# from .retrieving_idx import get_embadded
from .lexical_retrieval import LexicalSearch
from .generator import Generator
from colorama import Fore
from .cli import RagCli
import fire


def main():
    generate = Generator("output.json")
    # model_emb = get_embadded("all-MiniLM-L6-v2")  # vector sementique search
    lexical_retrive = LexicalSearch(vllm_path="vllm-0.10.1")
    try:
        rag_cli = RagCli()
        generate.generate_json(list(range(50)))
        fire.Fire(rag_cli)
        lexical_retrive.chunk_vllm(max_chunk_size=2000)
    except Exception as e:
        print(f'{Fore.RED}[ERROR] {e}')


if __name__ == "__main__":
    main()
