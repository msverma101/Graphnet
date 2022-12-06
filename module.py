from os import name
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from normalise import Normalizer
from common import NodeType, batch_index_select, batch_kdtree, device_common, to_device, batch_topk
from encode_process_decode import EncodeProcessDecode
device= device_common()

class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, output_size, message_passing_steps=15):
        super(Model, self).__init__()

        self._normalizer = {
                            'output': Normalizer(name='output_normalizer', range=(0,1)),
                            'node' : Normalizer(name='node_normalizer', range=(0,1)),
                            'edge' : Normalizer(name='edge_normalizer', range=(0,1)),
                            'punch_edge': Normalizer(name='punch_edge_normalizer', range=(0,1)), 
                            'die_edge':Normalizer(name='die_edge_normalizer', range=(0,1)),
                            'holder_edge':Normalizer(name='holder_edge_normalizer', range=(0,1))
                            }
        self.message_passing_steps = message_passing_steps
       
        self.learned_model = EncodeProcessDecode(
                                    output_size=output_size,
                                    latent_size=128,
                                    num_layers=2,
                                    message_passing_steps=self.message_passing_steps)

    def _build_graph(self, inputs, offset, is_training):
        """Builds input graph."""
        "world edges is list of edges between plate and punch, die, holder"
        world_edges = self.get_world_edges(inputs, offset, radius=1.5)

        # send inputs to device

        one_hot_node_type = F.one_hot(inputs['node_type'], NodeType.SIZE)
        node_features = torch.cat((one_hot_node_type, inputs['coords']), dim=-1)

        #########################Mesh Edges##################################
        mesh_edges= self.get_edge_set(inputs['coords'], inputs['edges'], name="edge")
        ########################World Edges##################################
        world_edges = self.get_world_edgeset(inputs, world_edges)
        
        node_normalizer = self._normalizer['node']
        return {'node_features': node_normalizer(node_features, is_training),
                          'edge_sets': [mesh_edges, *world_edges]}

    def forward(self, inputs, offset={'plate': 0, 'punch': 4981, 'die': 11041, 'holder': 16735}, is_training=True):
        graph = self._build_graph(inputs, offset, is_training=is_training)
        if is_training:
            return self.learned_model(graph)
        else:
            return self._update(inputs, self.learned_model(graph))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""

        displacement = self._normalizer['output'].inverse(per_node_network_output)

        # integrate forward
        position = inputs['coords'] + displacement
        return position

    def get_output_normalizer(self):
        return self._normalizer['output']
    

    def get_world_edges(self, inputs, offset, radius=1.5):
        #Need to set the radius right for now taking it as 1.5
        indices= torch.arange(0, inputs['coords'].size(1), device= device)
        coords_plate= torch.index_select(inputs['coords'], dim=1, index= indices[offset['plate']:offset['punch']])
        coords_punch= torch.index_select(inputs['coords'], dim=1, index= indices[offset['punch']:offset['die']])
        coords_die= torch.index_select(inputs['coords'], dim=1, index= indices[offset['die']:offset['holder']])
        coords_holder= torch.index_select(inputs['coords'], dim=1, index= indices[offset['holder']:])
        
        edges_punch= batch_topk(coords_plate, coords_punch, offset['punch'], radius=2.5)
        edges_die= batch_topk(coords_plate, coords_die, offset['die'], radius)
        edges_holder= batch_topk(coords_plate, coords_holder, offset['holder'], radius)
        return [edges_punch, edges_die, edges_holder]

    def get_world_edgeset(self, inputs, world_edges):
        edges_punch, edges_die, edges_holder = world_edges
        #######################Punch World Edges###############################
        
        
        edge_features_punch= torch.stack([inputs['friction'], inputs['depth']], dim=-1)
        edge_features_punch= torch.ones((1, edges_punch.size(-1), 1), device= device)*edge_features_punch[:, None, :] 
        
        punch_edges= self.get_edge_set(inputs['coords'], edges_punch, name="punch_edge", additional_edge_features=edge_features_punch)
        #########################Die World Edges###############################
        
        
        die_edges= self.get_edge_set(inputs['coords'], edges_die, name="die_edge")

        #######################Holder World Edges##############################

        
        edge_features_holder= torch.stack([inputs['friction'], inputs['F_press']], dim=-1)
        edge_features_holder= torch.ones((1, edges_holder.size(-1), 1), device= device)*edge_features_holder[:, None, :]
        
        holder_edges= self.get_edge_set(inputs['coords'], edges_holder, name="holder_edge", additional_edge_features=edge_features_holder)

        world_edges= [punch_edges, die_edges, holder_edges]
        return world_edges
    
    def get_edge_set(self, coords, edges, name="edge", additional_edge_features=None, is_training=True):
        edges.transpose(0,1).shape
        exit
        senders, receivers = edges.transpose(0,1)
        relative_world_pos = batch_index_select(coords, senders) - batch_index_select(coords, receivers)
        
        edge_features = torch.cat((
                            relative_world_pos,
                            torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)
        
        if additional_edge_features is not None:
            edge_features= torch.cat((edge_features, additional_edge_features), dim=-1)

        edge_normalizer= self._normalizer[name]
        return {'name':name,
                'features':edge_normalizer(edge_features, is_training),
                'receivers':receivers,
                'senders':senders}

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._normalizer['output'],path + "_output_normalizer.pth")
        torch.save(self._normalizer['edge'],path + "_edge_normalizer.pth")
        torch.save(self._normalizer['node'],path + "_node_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._normalizer['output']= torch.load(path + "_output_normalizer.pth")
        self._normalizer['edge']= torch.load(path + "_edge_normalizer.pth")
        self._normalizer['node']= torch.load(path + "_node_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()