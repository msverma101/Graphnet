import typing
from collections import namedtuple
import numpy as np
import torch
import webdataset as wds
from sklearn.preprocessing import MinMaxScaler
from data import Data


path= "/Users/parvezmohammed/ownCloud/Austausch_DeepLearning_Deepdrawing/01_Data/04_Kreuznapf_9857_Samples_6_Var"
dataset= Data(path)
l= len(dataset)
indices = torch.randperm(l)
batch_size= 500
i=0
while True:
    sink = wds.TarWriter(f"/Users/parvezmohammed/Downloads/Downloads/Fraunhofer/DeepLearning_Project/DeepLearning_Deepdrawing/data/data{i}.tar")

    for idx in range(batch_size):
        index= i*batch_size + idx
        if index==l:
            sink.close()
            exit() 
        input, output = dataset[indices[index]]
        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": input,
            "output.pyd": output,
        })
    sink.close()
    i+=1