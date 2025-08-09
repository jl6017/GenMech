import os
from PIL import Image
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from glob import glob
import random
random.seed(42)  # For reproducibility

class CurveMechPairDataset(Dataset):
    """Dataset that returns (curve_img, mech_img) given directory structure:
        root_dir/
            curve/   000001.png 000002.png ...
            mechanism/    000001.png 000002.png ...
    """
    def __init__(self, 
                 curve_files: list[str],
                 mech_files: list[str],
                 name_list: list[str],
                 transform=None):

        self.curve_files = curve_files
        self.mech_files = mech_files
        self.name_list = name_list
        default_tf = T.Compose([
            # T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),                        # [0,1]
        ])

        if transform is not None:
            self.transform = transform      
        else:
            self.transform = default_tf    

    def __len__(self):
        return len(self.curve_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        curve_img = Image.open(self.curve_files[idx]).convert("RGB")
        mech_img  = Image.open(self.mech_files[idx]).convert("RGB")

        curve_img = self.transform(curve_img)
        mech_img  = self.transform(mech_img)

        name = self.name_list[idx]
        return curve_img, mech_img, name


# ───────────────────────────────── Data loader helper ─────────────────────────────

def get_dataloader(batch_size: int = 32,
                   datapath: str = "./../dataset/",
                   dataset: list[str] = None,
                   train_ratio: float = 0.8,
                   num_workers: int = 4):

    tf = None
    curve_dir_list, mech_dir_list, name_list = [], [], []
    curve_subdir = "curve"
    mech_subdir = "mechanism"
    for d in dataset:
        curve_paths = sorted(glob(f"{datapath}/{d}/images/{curve_subdir}/*.png"))
        mech_paths = sorted(glob(f"{datapath}/{d}/images/{mech_subdir}/*.png"))

        curve_ids = [os.path.splitext(os.path.basename(p))[0] for p in curve_paths]
        mech_ids = [os.path.splitext(os.path.basename(p))[0] for p in mech_paths]
        assert curve_ids == mech_ids, f"Mismatch in {d}: curve vs mech file names"
        curve_dir_list.extend(curve_paths)
        mech_dir_list.extend(mech_paths)
        name_list.extend([f"{d}_{i}" for i in curve_ids])  # tri_2_0000, tri_2_0001, ...

    index_list = list(range(len(curve_dir_list)))
    random.shuffle(index_list)

    # data_size = min(len(index_list), 100000)  # Limit to 100k samples
    data_size = len(index_list)  # Use all available data
    train_size = int(data_size * train_ratio)
    train_indices = index_list[:train_size]
    val_indices = index_list[train_size:data_size]

    train_curve_files = [curve_dir_list[i] for i in train_indices]
    train_mech_files = [mech_dir_list[i] for i in train_indices]
    train_ids = [name_list[i] for i in train_indices]
    val_curve_files = [curve_dir_list[i] for i in val_indices]
    val_mech_files = [mech_dir_list[i] for i in val_indices]
    val_ids = [name_list[i] for i in val_indices]

    train_ds = CurveMechPairDataset(train_curve_files, train_mech_files, train_ids, transform=tf)
    val_ds   = CurveMechPairDataset(val_curve_files, val_mech_files, val_ids, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader
