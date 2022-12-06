
from common import loss_fn, to_device, visualise_geometry
from module import *
from data import *
from common import NodeType, loss_fn, device_common
from normalise import *
import torch
from torch.utils.data import DataLoader, random_split

import webdataset as wbs
import os

# torch.autograd.detect_anomaly= False
# torch.autograd.profiler.profile= False
# torch.autograd.gradcheck= False

device = device_common()


def load_dataset(data_files_dir, batch_size=1):
    """
    loading dataset using url with webdatset.

    Parameters
    ----------
    data_files_dir : String
        Path for the dataset.
    batch_size: int
        prefereably 1 since dataset is already batched. if not make create batch size
    """
    dataset             = wbs.WebDataset(data_files_dir).decode().to_tuple("input.pyd", "output.pyd")
    # dataset           = Data(data_files_dir)
    data_loader         = DataLoader(dataset, batch_size)#, pin_memory= True, num_workers= 2)
    return data_loader



def train(model, data_files_dir, epochs, sudo_epoch, offset, MSE_loss, visual_tensorboard = False):
    """
    Training is done with epochs and loss.

    Parameters
    ----------
    data_files_dir : String
        Path for the dataset.
    epoch : int
        
    """
    
    is_training = True
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    for k in range(0,1):
        # data_file =  data_files_dir + rf"\\data{k}.tar"
        data_file =  os.path.join(data_files_dir, f"data{k}.tar")
        train_loader = load_dataset(data_file)
        input, output = next(iter(train_loader))##here added just one file
        epoch_loss = []
        for epoch in range(1, epochs+1):
            output = to_device(output, device)
            network_output = model(input, offset, is_training)

            target_normalized = model.get_output_normalizer()(output['disps'][:, offset["plate"]: offset["punch"]])
            network_output = network_output[:, offset["plate"]: offset["punch"]]
            
            loss = MSE_loss(network_output, target_normalized)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if visual_tensorboard:
                learning_rate = scheduler.get_last_lr()[0]
                visual_tensorboard(loss.detach(), epoch, learning_rate, model, input)## extra functionaility to visulise

            print(f'Epoch [{epoch}/{epochs}], Train-Step-Loss: {loss.detach():.4f}')
            scheduler.step()

            epoch_loss.append(torch.tensor(loss.detach()).mean())
            # new_postion = input["coords"] + output["disps"]  ## extra functionaility to visulis
            # visualise_geometry(new_postion, network_output, offset) ## extra functionaility to visulise
            model.save_model("./models/")
            # model.save_model(r"models\\")

    train_loss= torch.tensor(epoch_loss).mean()
    return train_loss

def evaluate(model, data_files_dir, offset):
    
    is_training = True
    model.to(device)

    for i in range(1,2):
        # data_file =  data_files_dir + rf"\\data{i}.tar"
        data_file =  os.path.join(data_files_dir, f"data{i}.tar")
        val_loader = load_dataset(data_file)
        val_loss = []
        with torch.no_grad():
            model.evaluate()
            for sample, (input, output) in enumerate(val_loader):
                output= to_device(output, device)
                network_output = model(input, offset, is_training)
                loss = loss_fn(network_output, output, model, offset)
                val_loss.append(loss.detach())

                new_postion = input["coords"] + output["disps"]
                visualise_geometry(new_postion, network_output, offset)
    return val_loss




