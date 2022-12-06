import enum
import shutil
import numpy as np
from scipy.spatial import cKDTree
import numpy as np
import torch
from collections import namedtuple
from torch_scatter import scatter_mean, scatter_sum
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import os

# EdgeSet = {'name': None, 'features':None, 'senders':None, 'receivers':None}
# MultiGraph = {'node_features':None, 'edge_sets':None}



def device_common():
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device is assigned as CUDA')
  else:
    device = torch.device("cpu")   
    print('device is assigned as CPU')
  return device

class NodeType(enum.IntEnum):
  PLATE = 0
  PUNCH = 1
  DIE = 2
  HOLDER = 3
  SIZE = 4

def quads_to_edges(connectivity):
  edges = set()
  # Fix me for time
  # here an assumption is made that for a quadratic element with [i0, i1, i2, i3] nodes 
  # the edges are e1= [i0, i1], e2 = [i1, i2], e3= [i2, i3], e4= [i3, i0]
  # I have verified this for Kreuznapf01

  for (i0, i1, i2, i3) in connectivity:
      edges.add((i0, i1)) if(i0 < i1) else edges.add((i1, i0))
      edges.add((i1, i2)) if(i1 < i2) else edges.add((i2, i1))
      edges.add((i2, i3)) if(i2 < i3) else edges.add((i3, i2))
      edges.add((i3, i0)) if(i3 < i0) else edges.add((i0, i3))
  edges= sorted(edges)
  edges= np.array(edges)
  
  "because bidirectional"
  return np.vstack((edges, np.roll(edges, 1, axis=1) )).T

def quads_to_edges_new(connectivity):
  #Not so intuitive but faster code
  i0, i1, i2, i3 = connectivity.T
  ###########################i0<i1####################################
  id_01= i0<i1
  r0=np.where(id_01, i0, i1)
  r1= np.where(~id_01, i0, i1)
  ###########################i1<i2####################################
  id_12= i1<i2
  r0= np.r_[r0, np.where(id_12, i1, i2)]
  r1= np.r_[r1, np.where(~id_12, i1, i2)]
  ###########################i2<i3####################################
  id_23= i2<i3
  r0= np.r_[r0, np.where(id_23, i2, i3)]
  r1= np.r_[r1, np.where(~id_23, i2, i3)]
  ###########################i3<i0####################################
  id_30= i3<i0
  r0= np.r_[r0, np.where(id_30, i3, i0)]
  r1= np.r_[r1, np.where(~id_30, i3, i0)]
  ####################################################################

  edges= np.vstack((r0, r1))
  edges= np.unique(edges, axis=1)

  "because bidirectional"
  edges= np.hstack((edges, np.roll(edges,1, axis=0)))
  return edges

def kdtree_nearest_graph(coord_plate, coord_item, radius=1.5, offset=0):
  '''
  Creates edges between the nearest neighbours with the given radius
  Coord_plate: float
                coordinated of the plate
  coord_item: float
              coordinates of the external tool in contact
  radius : float
          
  '''
  tree_punch = cKDTree(coord_item)
  index = tree_punch.query_ball_point(coord_plate, radius)

  edges= [np.c_[[i]*len(idx),idx] for i, idx in enumerate(index) if idx ]
  edges= np.vstack(edges) 
  edges+= [[0, offset]]
  edges= torch.from_numpy(edges)
  return edges.T

def topk(coord_plate, coord_item, radius=1.5, offset=0):

  dist2= torch.norm(coord_item.unsqueeze(1) - coord_plate.unsqueeze(0), dim=2, p=None)

  knn2 = dist2.topk(100, largest=False, dim=0)
  indices= (knn2.values<=radius)

  edges=[]
  for i in range(coord_plate.size(0)):
    #i-> index of plate, idx-> index of item
    idx= knn2.indices[:,i][indices[:,i]]
    if len(idx)!=0:
      edges.append(torch.vstack((torch.ones_like(idx)*i, idx)))
  edges= torch.hstack(edges) 
  edges[1]+= offset 
  return edges

def batch_topk(batched_coord_plate, batched_coord_item, offset, radius=1.5):
  batch_size= batched_coord_plate.size(0)
  output= [topk(batched_coord_plate[i], batched_coord_item[i], radius=radius, offset=offset) for i in range(batch_size)]
  return torch.stack(output, dim=0)
  
def batch_index_select(batched_data, batched_indices, batch_dim=0):
  # batched_indices= batched_indices.long()
  batch_size= batched_data.size(batch_dim)
  output= [batched_data[i, batched_indices[i]] for i in range(batch_size)]
  return torch.stack(output, dim=batch_dim)

def batch_kdtree(batched_coord_plate, batched_coord_item, offset=0, radius=1.5):
  batch_size= batched_coord_plate.size(0)
  output= [kdtree_nearest_graph(batched_coord_plate[i], batched_coord_item[i], radius=radius, offset=offset) for i in range(batch_size)]
  return torch.stack(output, dim=0).long()

def batch_scatter(batched_data, batched_indices, reduction='sum', dim_size=None):
  scatter= {'sum': scatter_sum, 'mean': scatter_mean}
  batch_size= batched_data.size(0)
  batch_output= [scatter[reduction](batched_data[i], batched_indices[i], dim=0, dim_size=dim_size) for i in range(batch_size)]
  return torch.stack(batch_output, dim=0)

def loss_fn(network_output, output, model, offset, concentration_weight=50.0):
  """
  Loss function similar to mean squared error   
      
  """
  target_normalized = model.get_output_normalizer()(output['disps'])
  # build loss
  error = torch.sum((target_normalized - network_output) ** 2, dim=-1)
  error[:, offset["plate"]: offset["punch"]]*= concentration_weight
  loss = torch.mean(error)

  return loss

  
# plot parts
def visualise_geometry(target, prediction, offset):


  fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scatter3d", "rowspan": 2}, {"type": "scatter3d"}],
                                              [            None                    , {"type": "scatter3d"}]])
  
  prediction = prediction[:, offset["plate"]: offset["punch"]].detach().cpu().numpy()
  x = prediction[0, :, 0]
  y = prediction[0, :, 1]
  z = prediction[0, :, 2]

  fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1)), row=1, col=1)

  target = target[:, offset["plate"]: offset["punch"]].detach().cpu().numpy()
  x = target[0, :, 0]
  y = target[0, :, 1]
  z = target[0, :, 2]
                                  
  fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1)), row=1, col=2)

  difference = target - prediction
  x = difference[0, :, 0]
  y = difference[0, :, 1]
  z = difference[0, :, 2]

  fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1)), row=2, col=2)

  fig.update_layout(title='Difference', autosize=True, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
  fig.show()


def to_device(d:dict, device):
  for key, value in d.items():
    d[key]= value.to(device)
  return d


class LaunchTensorboard:
    
  """
  Parameters necessary for Launching tensorboard.

  Parameters
  ----------
  log_dir: string
        path for the log_dir folder where the logs of tensorboard are stored
        and visualised

  clear_on_exit : bool
      If True Clears the log_dir on exit and kills the tensorboard app.
  ----------

  """

  def __init__(self, log_dir, clear_on_exit=False):

    self.tb = program.TensorBoard()
    self.log_dir = log_dir
    self.clear_on_exit = clear_on_exit
    self.url = None
    self.tensor_board = SummaryWriter(self.log_dir)

    #summary_writer tensor board -- to be done
    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
    self.tensor_board = SummaryWriter(self.log_dir)

  def __call__(self, loss, epoch,learning_rate, model, input):
    # Set the path to log files
    self.tensor_board.add_scalar("Loss", loss, epoch)
    self.tensor_board.add_scalar("Learning rate",learning_rate, epoch)
    self.tensor_board.add_graph(model,input, verbose=True)
    for name, param in model.named_parameters():
      self.tensor_board.add_histogram(name,param,epoch)
      self.tensor_board.add_histogram(f'{name}.grad',param.grad,epoch)

    self.tensor_board.close() #method to make sure that all pending events have been written to disk


  def run(self):
    self.tb.configure(argv=[None, '--logdir', self.log_dir])
    print("Launching Tensorboard ")
    self.url = self.tb.launch()
    print(self.url)
    webbrowser.open_new_tab(self.url)
    try:
      input("Enter any key to exit")
    except:
      pass
    finally:
      if self.clear_on_exit:
        shutil.rmtree(self.log_dir, ignore_errors=True)
      print("\nCleared Logdir")
