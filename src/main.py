from cli import RagCli
import fire

def main():
    rag_cli = RagCli()
    print("Hello from rag!")
    fire.Fire(rag_cli)


if __name__ == "__main__":
    main()
