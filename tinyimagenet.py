import glob
import os
from zipfile import ZipFile

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io.image import ImageReadMode, decode_image, read_image

id_dict = {}
for i, line in enumerate(open("data/TinyImageNet/wnids.txt", "r")):
    id_dict[line.replace("\n", "")] = i


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("data/TinyImageNet/train/*/images/*.JPEG")
        self.transform = transform
        self.id_dict = id_dict
        self.targets = [
            self.id_dict[img_path.split("/")[3]] for img_path in self.filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.id_dict[img_path.split("/")[3]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("data/TinyImageNet/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id_dict
        self.cls_dic = {}
        for i, line in enumerate(
            open("data/TinyImageNet/val/val_annotations.txt", "r")
        ):
            a = line.split("\t")
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.cls_dic[img_path.split("/")[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


class CorruptTinyImageNetDataset(Dataset):
    def __init__(self, intensity, transform=None, corruption_name=None):
        if corruption_name is None:
            self.filenames = glob.glob(
                f"data/TinyImageNet/corrupt/*/{intensity}/*/*.JPEG"
            )
        else:
            self.filenames = glob.glob(
                f"data/TinyImageNet/corrupt/{corruption_name}/{intensity}/*/*.JPEG"
            )
        self.transform = transform
        self.id_dict = id_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.id_dict[img_path.split("/")[-2]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label
