import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # root_dir와 dataframe의 Path 컬럼을 결합하여 전체 경로 생성
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx, 1]
        label = int(label)  # Assuming label is already encoded to integer

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataset_dataframe(dataset_path):
    data = {'Path': [], 'Class': []}

    entries = os.listdir(dataset_path)

    for entry in entries:
        full_path = os.path.join(dataset_path, entry)
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            for file in files:
                file_path = os.path.join(entry, file)

                data['Path'].append(os.path.join(entry, file))
                data['Class'].append(entry)

    df = pd.DataFrame(data)
    return df