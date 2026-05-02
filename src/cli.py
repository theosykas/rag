from pydantic import BaseModel
from typing import List, Dict, Any
import time


class RagRes(BaseModel):
    text: str
    score: float = 0.0

class RagCli(BaseModel):
    pass

    def index(self, path_file: str) -> str:
        return f"index of file: {path_file} terminated"

    def search(self, single_qwery: str) -> List[str]:
        pass

    def search_dataset(self, search_res: str) -> List[List[str]]:
        pass

    def awnser(self, response: str) -> str:
        # prend question return final reponse
        return f"generated response {response}"

    def awnser_dataset(self, generate_response: List[str]) -> List[RagRes]:
        pass

    def evaluate_res(self, data_ref: str) -> Dict[str, Any]:
        pass
