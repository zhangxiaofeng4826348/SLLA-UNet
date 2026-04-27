# dataset_ssl.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms as T

class SSLDataset(Dataset):
    def __init__(self, image_folder, mask_folder=None, transform1=None, transform2=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_list = sorted(os.listdir(image_folder))
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.image_list)

    # dataset_ssl.py
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_list[idx])
        img = Image.open(img_path).convert("RGB")


        mask_tensor = torch.zeros(1, img.size[1], img.size[0])


        if self.transform1:
            img1, mask1 = self.transform1(img, mask_tensor)
        else:
            from torchvision import transforms as T
            img1 = T.ToTensor()(img)
            mask1 = mask_tensor

        if self.transform2:
            img2, mask2 = self.transform2(img, mask_tensor)
        else:
            from torchvision import transforms as T
            img2 = T.ToTensor()(img)
            mask2 = mask_tensor

        return {"image1": img1, "image2": img2, "mask1": mask1, "mask2": mask2}


