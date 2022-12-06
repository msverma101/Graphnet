import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from collections import namedtuple
from common import quads_to_edges_new, NodeType, quads_to_edges_new

# Input= namedtuple("Input", ["coords", "edges", "node_type", "F_press", "friction", "depth"]) #"prev_disp", "thickness"
# Output= namedtuple("Output", ["disps"])
        
class Data(Dataset):
    '''
        Interface class to load all the data and to call geomdata class to process it
    '''
    def __init__(self, data_path):
        super().__init__()
        self.data_path= data_path
        '''
        node_element_id is a connectivity matrix but with an offset 14693
        '''
        neid_plate = np.load(data_path+"/node_element_id_81000001.npy")
        neid_plate-= neid_plate.min()
        
        self.edges= self.get_edges(neid_plate) #, neid_punch, neid_die, neid_holder)

        self.experiment_data = pd.read_csv(data_path+"/Experiments_1_without_Error_Term.csv", sep=';', decimal=',')
        self.experiment_data = self.experiment_data[['F_press','fs','h_BT']].values.astype(np.float32)
        self.experiment_data = torch.tensor(self.experiment_data)
        self.n_exps= len(self.experiment_data)
       
        # self.data = GeomData(node_element_id, coords_plate, coords_punch, coords_holder, coords_die, experiment_data, disps_plate)
        ####################################################################
        # write this code in forward fucntion if it doesn't fit in the memory 
        # remove them from self and put indexing on them
        self.coords_plate = np.load(self.data_path+"/coord_all_81000001.npy").astype(np.float32)
        self.coords_punch = np.load(self.data_path+ "/coord_all_11000001.npy").astype(np.float32) 
        self.coords_die = np.load(self.data_path+ "/coord_all_1.npy").astype(np.float32)
        self.coords_holder = np.load(self.data_path+ "/coord_all_21000001.npy").astype(np.float32)
        ####################################################################
        self.t_nodes_plate= self.coords_plate.shape[1]
        self.t_nodes_punch= self.coords_punch.shape[1]
        self.t_nodes_die= self.coords_die.shape[1]
        self.t_nodes_holder= self.coords_holder.shape[1] 
        self.t_nodes= self.t_nodes_plate + self.t_nodes_punch + self.t_nodes_die + self.t_nodes_holder

        # print(f"offset_plate: {0}, offset_punch: {self.t_nodes_plate}, offset_die: {self.t_nodes_plate+self.t_nodes_punch}, offset_holder: {self.t_nodes_plate+self.t_nodes_punch+self.t_nodes_die}")
        
        self.node_type= torch.empty(self.t_nodes, dtype=torch.int64)
        self.node_type[0:self.t_nodes_plate]=NodeType.PLATE
        self.node_type[self.t_nodes_plate : self.t_nodes_punch+self.t_nodes_plate]= NodeType.PUNCH
        self.node_type[self.t_nodes_punch+self.t_nodes_plate : self.t_nodes_die+self.t_nodes_punch+self.t_nodes_plate]= NodeType.DIE
        self.node_type[self.t_nodes_die+self.t_nodes_punch+self.t_nodes_plate : self.t_nodes_holder+self.t_nodes_die+self.t_nodes_punch+self.t_nodes_plate]= NodeType.HOLDER

        ####################################################################
        # write this code in forward fucntion if it doesn't fit in the memory 
        # remove them from self and put indexing on them
        self.disps_plate = np.load(self.data_path+"/disp_all_81000001.npy").astype(np.float32)
        self.disps_punch = np.load(self.data_path+ "/disp_all_11000001.npy").astype(np.float32)
        self.disps_die = np.load(self.data_path+ "/disp_all_1.npy").astype(np.float32)
        self.disps_holder = np.load(self.data_path+ "/disp_all_21000001.npy").astype(np.float32)
        ####################################################################


    def __len__(self):
        return self.n_exps
    
    def __getitem__(self, idx):
        

        coords= np.vstack((self.coords_plate[idx], 
                           self.coords_punch[idx] + np.array([0,0,1.505], dtype=np.float32), 
                           self.coords_die[idx] + np.array([0,0, -10.5], dtype=np.float32), 
                           self.coords_holder[idx] + np.array([0,0,0.505], dtype=np.float32)))
        
        coords= torch.tensor(coords)
        disps= np.vstack((self.disps_plate[idx], self.disps_punch[idx], self.disps_die[idx], self.disps_holder[idx]))
        disps= torch.tensor(disps)

        F_press, friction, depth= self.experiment_data[idx]

        inputs= {'coords':coords, 'edges':self.edges, 'node_type': self.node_type, 'F_press':F_press, 'friction':friction, 'depth':depth}
        outputs= {'disps':disps}
        return (inputs, outputs)
    
    def get_edges(self, neid_plate): #, neid_punch, neid_die, neid_holder
        "If edges have to be redefined for every forward call"

        edges_plate= quads_to_edges_new(neid_plate)
        # edges_punch= quads_to_edges_new(neid_punch)
        # edges_die= quads_to_edges_new(neid_die)
        # edges_holder= quads_to_edges_new(neid_holder)

        # edges= np.hstack((edges_plate, edges_punch, edges_die, edges_holder))
        edges= edges_plate
        return edges

        


