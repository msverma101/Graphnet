import torch
from common import device_common, to_device
from data import Data
from data_dir import data_files_dir, log_dir
from module import Model
from torch.utils.data import DataLoader
import webdataset as wbs
# from normalise import Normalizer
# from pytorch_model_summary import summary

# data_files_dir = "../data/data0.tar"

# dataset = wbs.WebDataset(data_files_dir).decode().to_tuple("input.pyd", "output.pyd")
# data_loader    = DataLoader(dataset, batch_size=1)#, num_workers=0, pin_memory=True)
# device= device_common()
offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}


data_files_dir = "/home/snehaverma/Downloads/Fraunhofer/Deep/data_dict/data0.tar"

dataset = wbs.WebDataset(data_files_dir).decode().to_tuple("input.pyd", "output.pyd")
data_loader    = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
device= device_common()
output_size = 3
visual_tensorboard = LaunchTensorboard(log_dir = "./log_dir/log")
model = Model(output_size, message_passing_steps=2)
for inputs, outputs in data_loader:
    inputs= to_device(inputs, device)
    outputs= to_device(outputs, device)

    exit()
