import os
from PIL import Image

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