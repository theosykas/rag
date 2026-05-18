from colorama import Fore
from .cli import RagCli
import fire


def main() -> None:
    """. Entry point for the CLI application orchestrating the RAG execution
    framework.

        Initializes the core command-line wrapper instance targeting the
        localized
        vLLM repository codebase and hands over control to the Python Fire
        subsystem
        to dynamically parse terminal arguments, routing executions safely
        through a
        top-level exception monitoring block.

        Returns:
            None: Orchestrates process execution and handles CLI dispatching.

        Raises:
            Exception: Traps any initialization or system runtime errors,
            printing
                formatted telemetry logs to stderr without halting the console
                environment.
        """
    try:
        rag_cli = RagCli("data/raw/vllm-0.10.1")
        fire.Fire(rag_cli)
    except Exception as e:
        print(f'{Fore.RED}[ERROR] {e}')


if __name__ == "__main__":
    main()
