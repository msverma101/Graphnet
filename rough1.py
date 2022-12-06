import torch
from run_model import train, evaluate
from common import LaunchTensorboard
from module import Model
from data_dir import data_files_dir, log_dir
torch.set_printoptions(profile='default')
torch.cuda.empty_cache()
import time


offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}
epochs = 500
output_size = 3


model = Model(output_size, message_passing_steps=1)
# model.load_model(r"..\\model_500")
model.load_model("/home/snehaverma/Downloads/Fraunhofer/Deep/DeepLearning_Deepdrawing/graphnet/model_222/")

evaluate(model, data_files_dir, offset, sample_limit=100)



