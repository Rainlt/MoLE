from torch.utils.data import Dataset
import os
import random
import copy
import json
from PIL import Image
import torch
prompt = "{question}\nAnswer the question using a single word or phrase."

class HallusionBenchDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        base_path: str,
        require_cot: bool=False,
        trans = None
    ):
        self.path = json_path
        self.meta = [item for item in json.load(open(json_path)) if item['visual_input']=='1']
        self.base_path = base_path
        self.require_cot = require_cot
        self.trans = trans

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return 'yes' if self.meta[i]['gt_answer'] == '1' else 'no'

    def __getitem__(self, i: int):
        item = self.meta[i]
        question = item['question']
        img_fn = item['filename'].lstrip('./')
        label = torch.tensor(int(self.meta[i]['gt_answer']))#self.getlabel(i)
        image_path = os.path.join(self.base_path, img_fn)
        input_sent = prompt.format(question=(question + ' Please answer yes or no.'))

        image = Image.open(image_path).convert("RGB")
        if self.trans is not None:
            image = self.trans(image)


        return {'image': image, 'query': input_sent, 'label': label,"image_path": image_path}
