import torch
import torch.nn as nn
from common import device_common
from graphnet import EdgeModel, NodeModel, MetaLayer
from collections import OrderedDict
import functools
class LazyMLP(nn.Module):
    
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            if index < (num_layers - 1):
                self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
                self._layers_ordered_dict["relu_" + str(index)] = nn.LeakyReLU(0.2)
            else:
                self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size, bias=False)

        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        y = self.layers(input)
        return y

class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size, num_edge_sets=4):
        super().__init__()
        self._latent_size = latent_size
        self.node_model = make_mlp(latent_size)
        self.edge_models = nn.ModuleList([])

        for i in range(num_edge_sets):
          edge_model = make_mlp(latent_size)
          self.edge_models.append(edge_model)

    def forward(self, graph):
        node_latents = self.node_model(graph['node_features'])
        
        senders, receivers= [], []
        edge_features= []
        for index, edge_set in enumerate(graph['edge_sets']):
            feature = edge_set['features']
            latent = self.edge_models[index](feature)

            if index==0: # mesh edge
                mesh_edge= edge_set
                mesh_edge['name']= 'mesh_edge'
                mesh_edge['features']= latent

            else: # world edges
                edge_features.append(latent)
                senders.append(edge_set['senders'])
                receivers.append(edge_set['receivers'])
            
        
        senders, receivers= torch.hstack(senders), torch.hstack(receivers)
        edge_features= torch.cat(edge_features, dim=1)
        world_edge=  {'name': 'world_edge', 
                    'features': edge_features,
                    'receivers': receivers,
                    'senders': senders}


        graph['node_features']= node_latents
        graph['edge_sets']= [mesh_edge, world_edge]

        return graph


class Decoder(nn.Module):
    """Decodes node features from graph."""
    # decoder = self._make_mlp(self._output_size, layer_norm=False)
    # return decoder(graph['node_features'])

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        return self.model(graph['node_features'])


class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection
    (features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent 
    graph will be feed into original processor
    message_passing_steps - epochs
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps):
        super().__init__()
        self.graphnet_blocks = nn.ModuleList()

        for index in range(message_passing_steps):
            self.graphnet_blocks.append(MetaLayer(EdgeModel(make_mlp, output_size),
                                                  NodeModel(make_mlp, output_size))
                                        )

    def forward(self, latent_graph):
        for graphnet_block in self.graphnet_blocks:
            latent_graph = graphnet_block(latent_graph)
        return latent_graph


class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_steps):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps

    

        self.encoder = Encoder(make_mlp=self._make_mlp,
                               latent_size=self._latent_size)

        self.processor = Processor(make_mlp=self._make_mlp,
                                   output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)


    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * (self._num_layers - 1) + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""

        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)


        return self.decoder(latent_graph)

