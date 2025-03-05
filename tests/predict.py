"""
Script to predict the view and laterality of a unlabeled Mammography

IMG_FOLDER -> Path to folder with images,
    no sub folders takes Dicom, JPEG

"""
# Python imports
from typing import Literal, Tuple, List, Set, Union, Callable
from enum import Enum
import os, sys

# Torch imports
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Other
import numpy as np, pandas as pd
import pydicom
from PIL import Image

# Utils
class Color(Enum):
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    def __call__(self, text: str, **kwargs) -> None:

        print(self.value + text + Color.RESET.value, **kwargs)
    
    def apply(self, text: str) -> None:

        return self.value + text + Color.RESET.value

# Settings
MODEL_PATH: str = "models/SAMMY.pt"
IMG_FOLDER: str = "INBreast/ALL-IMGS"                                         # <--------- Replace with your Folder for images or list of image paths
DATA_CSV: str = pd.read_excel("INBreast/INbreast.xls", dtype=str)                        # <--------- Replace with your csv for labels pd.DataFrame
IMAGE_PATH_COL: str = "File Name"                                                # <--------- Column for associating images w entries
IMAGE_COL_FIND: Callable = lambda image_col_path, file: image_col_path == file.split("_")[0] # <--------- If Image Column not exactly file name, None if Image Col == File Name
VIEW_COL: str = "View"                                                           # <--------- Column to get label
MODE: Literal["predict", "evaluate"] = "evaluate"
SUPPORTED_FILE_TYPES: Set[str] = {"dcm", "jpeg", "jpg", "png", "webp"}
# raise NotImplementedError(Color.RED.apply("Make your changes here!!!"))

# Checks that Evaluate mode can find labels
assert MODE != "evaluate" or not (DATA_CSV is None or IMAGE_PATH_COL is None or VIEW_COL is None), Color.RED.apply("Evaluate Mode cannot run labels DF not fully informed")

print(
Color.BLUE.apply(f"""
In {MODE.capitalize()} Mode
Image Folder {IMG_FOLDER}
Labels CSV \n{DATA_CSV[[IMAGE_PATH_COL, VIEW_COL]].head(5)}
"""
))

# Data processing functions

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda i: i / i.max()),
])

def load_image(img_path: str, skip_missing: bool = False) -> Union[torch.Tensor, None]:
    """Loads and processes images as Torch Tensors \n
    Supported File Types: \n
    Dicom (.dcm) \n
    JPEG (.jpg | .jpeg) \n
    PNG (.png), Webp (.webp) \n

    Args:
        img_path (str): Path to Image
        skip_missing (bool, optional): If image not found returns None, does not raise FileNotFoundError. Defaults to False.

    Raises:
        FileNotFoundError: If Image path does not exist and not skip_missing
        ValueError: If file type is not supported

    Returns:
        torch.Tensor: Resized, Grayscaled, and Normalized Image
    """
    # Checks if path is valid
    if not os.path.exists(img_path):

        # Return none do not raise Error
        if skip_missing:
            return None

        raise FileNotFoundError(Color.RED.apply(f"Image File {img_path} not found!"))
    
    # Find and load file type
    file_type: str = img_path.split(".")[-1].lower()

    match file_type:

        # Dicom
        case "dcm":

            return transform(
                Image.fromarray(pydicom.dcmread(img_path).pixel_array)
            )
        
        case "jpg" | "jpeg" | "png" | "webp":

            return transform(
                Image.open(img_path).load()
            )
        
        case _:

            raise ValueError(Color.RED.apply(f"File format .{file_type} is not supported. Try one of these {SUPPORTED_FILE_TYPES}"))

# Custom Dataset functionality for Quicker Prediction
class EagerLoader(Dataset):

    def __init__(self, path: Union[str, List[str]] = IMG_FOLDER) -> None:
        """Loads Images Eagerly, saved to memory \n
        Slower init Faster use, high memory overhead

        Args:
            path (Union[str, List[str]]): Either string path to directory of images or list of paths to individual images
        """

        self.path = path

        # Load Dataset to self.images
        self.load()

        return

    def load(self) -> None:
        """Iterates over files in directory and load images"""

        images = []
        labels = []

        self.img_paths_ = os.listdir(self.path) if isinstance(self.path, str) else self.path

        l = self.img_paths_.__len__()

        for i, ipath in enumerate(self.img_paths_):

            update: str = f"{i+1}/{l}\r"

            match (i+1) / l:

                case x if x < 0.33:
                    Color.RED(update, end='')

                case x if x < 0.66:
                    Color.YELLOW(update, end='')

                case _:
                    Color.GREEN(update, end='')

            # Ignore documentation or csv files
            if ipath.split('.')[-1] not in SUPPORTED_FILE_TYPES:

                continue

            if MODE == "evaluate":

                found_row = DATA_CSV[DATA_CSV[IMAGE_PATH_COL].apply(str).apply(IMAGE_COL_FIND, args=(ipath,))] if IMAGE_COL_FIND is not None else DATA_CSV[DATA_CSV[IMAGE_PATH_COL] == ipath]

                if found_row.empty:

                    continue

                labels.append(
                    1 if \
                        found_row.iloc[0][VIEW_COL] == "MLO"
                    else 0
                )
            
            images.append(
                load_image(self.path + '/' + ipath).to(torch.float32)
            )

        self.images = torch.unsqueeze(torch.concat(images), -1)

        if MODE == "evaluate":

            self.labels = torch.unsqueeze(torch.Tensor(labels), -1)

        else:

            self.labels = torch.zeros(size=(l,1))
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        if MODE == "evaluate":

            return self.images[idx], self.labels[idx]
        
        # Not Eval
        return self.images[idx]

# Custom Dataset functionality for Less Memory Overhead
class LazyLoader(Dataset):

    def __init__(self, path: Union[str, List[str]] = IMG_FOLDER) -> None:
        """Loads Images Eagerly, saved to memory \n
        Slower init Faster use, high memory overhead

        Args:
            path (Union[str, List[str]]): Either string path to directory of images or list of paths to individual images
        """

        self.path = path

        # Load Paths to self.img_paths_
        self.load()

        return

    def load(self) -> None:
        """Saves list of paths to Images"""

        self.img_paths_ = os.listdir(self.path) if isinstance(self.path, str) else self.path
    
    def __len__(self):
        return self.img_paths_.__len__()

    def __getitem__(self, idx):

        path = self.img_paths_[idx]

        img = load_image(self.path + '/' + path)

        label = 0

        if MODE == "evaluate":
            label = 1 if \
                    DATA_CSV[
                        IMAGE_COL_FIND(DATA_CSV[IMAGE_PATH_COL], path) if IMAGE_COL_FIND else DATA_CSV[IMAGE_PATH_COL] == path
                    ][VIEW_COL] \
                    == "MLO" \
                else 0
            
        return img, label

def laterality(imgs: torch.Tensor) -> torch.Tensor:
    """[left, right]

    Args:
        imgs (torch.Tensor): _description_

    Returns:
        torch.Tensor: (x, 2) tensor where 0 is left and 1 is right
    """ 

    left = imgs[:, :, :112]
    right = imgs[:, :, 112:]

    left_sum = left.sum((1, 2))
    right_sum = right.sum((1, 2))

    print(left_sum.shape)

    return torch.stack([left_sum, right_sum], dim=1) / imgs.sum((1,2))

class SAMMY(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        """
        Returns Logits
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=-1)

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Dataset
    #data = EagerLoader()

    # Create Dataloader
    #data_loader = DataLoader(data, batch_size=32)

    # Load Model
    model = SAMMY()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    model.eval()
    model.to(device)

    


