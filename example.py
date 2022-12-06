import torch
from run_model import train, evaluate
from common import LaunchTensorboard
from module import Model
from data_dir import data_files_dir, log_dir
import time

##############################
#    Run model
##############################

print("file running run_model ")
offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}
epochs = 10

output_size = 3


model = Model(output_size, message_passing_steps=6)
# visual_tensorboard = LaunchTensorboard(log_dir = "./log_dir")


start = time.time()
train(model, data_files_dir, epochs, offset)
print('time:', time.time()-start)

evaluate(model, data_files_dir, offset)
# for param in model.parameters():
#     print("here is model",model)

# visual_tensorboard.run()

