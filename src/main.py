from .generator import Generator
from colorama import Fore
from .cli import RagCli
import fire
# import sys


def main():
    generate = Generator("output.json")
    try:
        rag_cli = RagCli()
        generate.generate_json(list(range(50)))
        fire.Fire(rag_cli)
    except Exception as e:
        print(f'{Fore.RED}[ERROR] {e}')


if __name__ == "__main__":
    main()
