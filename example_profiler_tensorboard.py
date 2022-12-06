import torch.utils.benchmark as benchmark

import torch
from run_model import train, evaluate
from common import LaunchTensorboard
from module import Model
from data_dir import data_files_dir, log_dir
import torch.profiler
torch.set_printoptions(profile='full')
torch.cuda.empty_cache()




prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)
prof.start()
##############################
#    Run model
##############################

print("file running run_model ")
offset = {'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}
epochs = 1
output_size = 3

model = Model(output_size, message_passing_steps=2)
train(model, data_files_dir, epochs, offset, prof, visual_tensorboard = True)


#############################
prof.stop()


