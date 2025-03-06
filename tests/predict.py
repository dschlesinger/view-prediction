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
from torchmetrics import classification
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

    def __call__(self, text: str, end:str='', **kwargs) -> None:

        print(self.value + text + Color.RESET.value, end=end, **kwargs)
    
    def apply(self, text: str) -> None:

        return self.value + text + Color.RESET.value

def color_score(s: float):
    """Assumes 0 -> 1 Scale, returns colored text"""

    match s:

        case x if x < 0.33:
            return Color.RED.apply

        case x if x < 0.66:
            return Color.YELLOW.apply

        case _:
            return Color.GREEN.apply

# Settings
MODEL_PATH: str = "models/SAMMY.pt"
IMG_FOLDER: str = "INBreast/ALL-IMGS"                                         # <--------- Replace with your Folder for images or list of image paths
DATA_CSV: str = pd.read_excel("INBreast/INbreast.xls", dtype=str)                        # <--------- Replace with your csv for labels pd.DataFrame
IMAGE_PATH_COL: str = "File Name"                                                # <--------- Column for associating images w entries
IMAGE_COL_FIND: Callable = lambda image_col_path, file: image_col_path == file.split("_")[0] # <--------- If Image Column not exactly file name, None if Image Col == File Name
VIEW_COL: str = "View"                                                           # <--------- Column to get label
MODE: Literal["predict", "evaluate"] = "predict"
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

        self.img_paths_ = list(filter(lambda x: x.split(".")[-1] in SUPPORTED_FILE_TYPES, os.listdir(self.path))) if isinstance(self.path, str) else self.path

        l = self.img_paths_.__len__()

        for i, ipath in enumerate(self.img_paths_):

            update: str = f"{i+1}/{l}\r"

            print(
                color_score((i+1)/l)(update),
                end=''
            )

            # Ignore documentation or csv files
            if ipath.split('.')[-1] not in SUPPORTED_FILE_TYPES:

                continue

            if MODE == "evaluate":

                found_row = DATA_CSV[DATA_CSV[IMAGE_PATH_COL].apply(str).apply(IMAGE_COL_FIND, args=(ipath,))] if IMAGE_COL_FIND is not None else DATA_CSV[DATA_CSV[IMAGE_PATH_COL] == ipath]

                if found_row.empty:

                    continue

                labels.append(
                    [0, 1] if \
                        found_row.iloc[0][VIEW_COL] == "MLO"
                    else [1, 0]
                )

            else:

                labels.append(None)
            
            images.append(
                load_image(self.path + '/' + ipath).to(torch.float32)
            )

        self.images = torch.unsqueeze(torch.concat(images), -1).permute((0,3,1,2))

        if MODE == "evaluate":

            self.labels = torch.unsqueeze(torch.Tensor(labels), -1)

        else:

            self.labels = torch.full(fill_value=-1, size=(l,2))
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        return torch.Tensor(self.images[idx]), torch.Tensor(self.labels[idx])

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

        self.img_paths_ = list(filter(lambda x: x.split(".")[-1] in SUPPORTED_FILE_TYPES, os.listdir(self.path))) if isinstance(self.path, str) else self.path
    
    def __len__(self):
        return self.img_paths_.__len__()

    def __getitem__(self, idx):

        path = self.img_paths_[idx]

        img = load_image(self.path + '/' + path).to(torch.float32)

        label = [-1, -1]

        if MODE == "evaluate":
            label = [0, 1] if \
                    DATA_CSV[
                        IMAGE_COL_FIND(DATA_CSV[IMAGE_PATH_COL], path) if IMAGE_COL_FIND else DATA_CSV[IMAGE_PATH_COL] == path
                    ][VIEW_COL] \
                    == "MLO" \
                else [1, 0]
            
        return img, label

# Function to predict laterality
def laterality(imgs: torch.Tensor) -> torch.Tensor:
    """[left, right]

    Args:
        imgs (torch.Tensor): shape(x,1,224,224)

    Returns:
        torch.Tensor: (x, 2) tensor where 0 is left and 1 is right
    """ 

    left = imgs[:, :, :, :112]
    right = imgs[:, :, :, 112:]

    left_sum = left.sum((1, 2, 3))
    right_sum = right.sum((1, 2, 3))

    sides = torch.stack([left_sum, right_sum], dim=1)
    total = imgs.sum((1,2,3)).unsqueeze(1)

    return sides / total

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

    print("Loading Data...")

    # Load Dataset, EagerLoader Prefered
    data = EagerLoader()

    # Create Dataloader
    data_loader = DataLoader(data, batch_size=32)

    print("Loading Model...")

    # Load Model
    model = SAMMY()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    print("Predicting View...")

    with torch.no_grad():
        num_batches = data_loader.__len__()

        for i, batch in enumerate(data_loader):

            update: str = f"{i+1}/{num_batches}\r"

            print(
                color_score((i+1)/num_batches)(update),
                end=''
            )

            x, y = batch 

            preds = model(x)
            
            y_pred.extend(preds.tolist()) 
            if MODE == "evaluate":
                y_true.extend(torch.argmax(y, dim=1).tolist())

    y_pred = torch.Tensor(y_pred).reshape((data.__len__(), 2))

    if MODE == "evaluate":

        y_true = torch.Tensor(y_true).reshape((data.__len__(),)).to(torch.long)

        metrics = [classification.Accuracy(task="multiclass", num_classes=2), classification.AUROC(task="multiclass", num_classes=2), torch.nn.CrossEntropyLoss()]

        for m in metrics:
            
            score = m(y_pred, y_true)

            print(
                Color.MAGENTA.apply(f"{m.__class__.__name__}: "),
                color_score(score)(f"{score:.4}")
            )
    
    else: # Predict

        # Save predictions

        Predictions = pd.DataFrame(
            columns={
                "File": str,
                "Laterality": str,
                "View": str,
            }
        )

        Predictions['File'] = pd.Series(data.img_paths_)

        Predictions['View'] = pd.Series(['CC' if yp == 0 else 'MLO' for yp in torch.argmax(y_pred, dim=1)])

        print("Predicting Laterality...")

        lat_pred = []

        # Get Laterallity
        for batch in data_loader:
            x, y = batch  # Unpack batch (if labels exist)

            lat = laterality(x)
            
            lat_pred.extend(torch.argmax(lat, dim=1).tolist())

        Predictions['Laterality'] = pd.Series(['L' if lp == 0 else 'R' for lp in lat_pred])

        print("Saving...")

        Predictions.to_csv('predictions.csv', index=False)
    


