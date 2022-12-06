import torch
import torch.nn as nn
from run_model import  load_dataset
from common import loss_fn
from torch_lr_finder import LRFinder
from module import Model
from data_dir import data_files_dir, log_dir
torch.set_printoptions(profile='full')
torch.cuda.empty_cache()
import os

##############################
#    Run model
##############################

print("file running run_model ")
offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}
epochs = 1
output_size = 3


model = Model(output_size, message_passing_steps=5)
for k in range(1,2):
        # data_file =  data_files_dir + rf"\\data{k}.tar"
        data_file =  os.path.join(data_files_dir, f"data{k}.tar")
        train_loader = load_dataset(data_file)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
criterion = nn.MSELoss()
lr_finder = LRFinder(model, optimizer, criterion)

lr_finder.range_test(train_loader, end_lr=100, num_iter=50)
lr_finder.reset() # to reset the model and optimizer to their initial state
ax, lr = lr_finder.plot() 


print("the learning rate is",lr)


