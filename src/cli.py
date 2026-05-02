from pydantic import BaseModel
from json import JSONDecodeError
from typing import List, Dict, Any
from .data_models import RagDataset
from colorama import Fore
import json


class RagRes(BaseModel):
    text: str
    score_k: float = 0.0


class RagCli(BaseModel):
    pass

    def index(self, path_file: str) -> str:
        return f"index of file: {path_file} terminated"

    def search(self, single_qwery: str) -> List[str]:
        pass

    def search_dataset(self, search_path: str, k: int, saving_output:
                       str = "output") -> None:
        try:
            with open(search_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            dataset = RagDataset(**dataset)
        except JSONDecodeError as e:
            print(f'{Fore.RED}[Error] {e}')
        return None

    def awnser(self, response: str) -> str:
        # prend question return final reponse
        return f"generated response {response}"

    def awnser_dataset(self, generate_response: List[str]) -> List[RagRes]:
        pass

    def evaluate_res(self, data_ref: str) -> Dict[str, Any]:
        pass
