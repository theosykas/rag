from json import JSONDecodeError
from typing import List
from tqdm import tqdm
import json


class Generator:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def generate_json(self, data_list: List) -> None:
        data = []
        try:
            for i in tqdm(data_list, ascii="---====►",
                          desc="Processing (json)", colour="green"):
                data.append(i)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                output = json.dump(data, f, indent=4)
                return output
        except JSONDecodeError as e:
            print(f'[Error] {e}')
        return None
