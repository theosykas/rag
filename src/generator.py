import json
from typing import List
from tqdm import tqdm
# import time


class Generator:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def generate_json(self, data_list: List) -> List[str]:
        data = []
        for i in tqdm(data_list, ascii="....::::####",
                      desc="Processing (json)", colour="green"):
            # time.sleep(0.1)
            data.append(i)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
