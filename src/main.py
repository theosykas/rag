from colorama import Fore
from .cli import RagCli
import fire


def main():
    try:
        rag_cli = RagCli("data/raw/vllm-0.10.1")
        fire.Fire(rag_cli)
    except Exception as e:
        print(f'{Fore.RED}[ERROR] {e}')


if __name__ == "__main__":
    main()
