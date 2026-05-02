from .generator import Generator
from .cli import RagCli
import fire


def main():
    rag_cli = RagCli()
    generate = Generator("output")
    try:
        generate.generate_json(list(range(50)))
        fire.Fire(rag_cli)
    except Exception as e:
        print(f'error {e}')


if __name__ == "__main__":
    main()
