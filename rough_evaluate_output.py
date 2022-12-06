import torch
from common import loss_fn, device_common, to_device
from run_model import evaluate, load_dataset

from module import Model
from data_dir import data_files_dir, log_dir
torch.set_printoptions(profile='default')
torch.cuda.empty_cache()
import os

device= device_common()

offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}
epochs = 500
output_size = 3


model = Model(output_size, message_passing_steps=1)
# model.load_model(r"..\\model_500")
model.load_model("/home/snehaverma/Downloads/Fraunhofer/Deep/DeepLearning_Deepdrawing/graphnet/model_300/")
is_training = True
data_file =  os.path.join(data_files_dir, f"data1.tar")
val_loader = load_dataset(data_file)
val_loss = []
with torch.no_grad():
    model.evaluate()
    for sample, (input, output) in enumerate(val_loader):
        output  = to_device(output, device)
        target_normalized = model.get_output_normalizer()(output['disps'])

        print("target_normalized disp shape",target_normalized.shape)
        print("target_normalized disp",target_normalized)

        # network_output = model(input, offset, is_training)
        # print("network_output disp shape",network_output.shape)
        # print("network_output disp",network_output)

        break
