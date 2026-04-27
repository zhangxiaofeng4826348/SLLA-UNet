from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.label_file = label_file
        self.transform = transform

        df = pd.read_excel(label_file)
        df["Name"] = df["Name"].astype(str).str.strip()
        df.set_index("Name", inplace=True)

        self.label_col = "Label"

        exclude_cols = [self.label_col]
        self.clin_cols = [c for c in df.columns if c not in exclude_cols]


        df[self.clin_cols] = df[self.clin_cols].astype(float)
        df[self.clin_cols] = df[self.clin_cols].fillna(df[self.clin_cols].mean())

        self.clin_mean = df[self.clin_cols].mean()
        self.clin_std  = df[self.clin_cols].std().replace(0, 1.0)  # 避免除 0

        self.df = df


        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        mask_file = image_file.replace(".png", ".nii")
        mask_path = os.path.join(self.mask_folder, mask_file)

        image = Image.open(image_path).convert("RGB")

        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask = sitk.GetArrayFromImage(mask_img)
        if mask.ndim == 3:
            mask = mask[0]

        if self.transform is not None:
            image = self.transform(image)  # [3,H,W]
            mask = torch.from_numpy(np.array(mask)).float()  # [H,W]

        # ---  mask  [1,H,W] ---
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        key = os.path.splitext(image_file)[0].strip()

        if key not in self.df.index:
            raise KeyError(
                f"{key}  {self.label_file} not found，"
            )

        row = self.df.loc[key]

        # 分类标签
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        # 临床向量
        clin_vals = row[self.clin_cols].astype(float)
        clin_norm = (clin_vals - self.clin_mean) / self.clin_std
        clin_vec = torch.tensor(clin_norm.values, dtype=torch.float32)  # [clin_dim]

        return {
            'image': image,  # [3,H,W]
            'mask': mask,  # [1,H,W]
            'label': label,  # 0/1
            'clin': clin_vec,  # [clin_dim]
            'filename': image_file
        }

