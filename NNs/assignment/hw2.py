from typing import Text, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

"""
This home work deals with:
1) Feed Forward Neural Networks.
2) Use Pytorch to design an entire project.
3) Build a `Dataset`, `DataLoader`, `Feed Forward Neural Networks`, `Trainer` with `train` and `evaluate` methods.
"""

"""
Dataset Class
"""

class FeedForwardDataset(Dataset):
    """
    You are required to complete this this skeleton code to read the four (.pt) files in the `data` folder.
    The four `.pt` files represent `train_input`, `train_output`, `test_input` and `test_output`.
    The data in the four files are pytorch tensors. The data is ready to be fed into your model.
    You are not required to preprocess, transform or perform any processing on the dataset.

    You will load the `.pt` files and use this class `FeedForwardDataset()` to construct two dataloaders (train_data_loader) and (test_dataloader)

    :params: input -> input data
    :params: output -> output data

    :returns: Pytorch Dataset class that can be used by the DataLoader

    For help: see the documentation -> https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, input: Text, output: Text) -> None:
        raise NotImplementedError
        # YOUR CODE

    def __len__(self) -> int:
        """This function just returns the length of the labels when called"""
        raise NotImplementedError
        # YOUR CODE

    def __getitem__(self, idx: int) -> Dict[Text, torch.Tensor]:
        """This function is used by Pytorch's Dataset module to get a sample and construct the dataset. 
        When initialised, it will loop through this function creating a sample from each instance in the dataset"""
        raise NotImplementedError
        # YOUR CODE

"""
Neural Networks consist of only 1 or 2 hidden layers.
"""

class DeepNN(nn.Module):
    """
    Neural Networks consist of 4 or 5 hidden.

    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError
        # YOUR CODE

    def forward(self, x):
        raise NotImplementedError
        # YOUR CODE



"""
Training
"""

class Training:
    def __init__(self, model, inputs, outputs) -> None:
        """add Docs"""
        raise NotImplementedError
        # YOUR CODE
        
    def train(self):
        """add Docs"""
        raise NotImplementedError
        # YOUR CODE



class Evaluation:
    def __init__(self) -> None:
        """add Docs"""
        raise NotImplementedError
        # YOUR CODE
    def eval(self):
        raise NotImplementedError
        # YOUR CODE

