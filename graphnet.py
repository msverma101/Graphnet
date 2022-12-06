import torch
import torch.nn as nn
from common import batch_index_select, batch_scatter


class EdgeModel(nn.Module):
    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self.edge_models= nn.ModuleDict({
                                            'mesh_edge': make_mlp(latent_size),
                                            'world_edge': make_mlp(latent_size)
                                        })
        
    def forward(self, graph):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_features: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        for index, edge_set in enumerate(graph['edge_sets']):

            feature = torch.cat((batch_index_select(graph['node_features'], edge_set['senders']), 
                                batch_index_select(graph['node_features'], edge_set['receivers']), 
                                edge_set['features']), dim=-1)

            feature = self.edge_models[edge_set['name']](feature)

            graph['edge_sets'][index]['features']+= feature 


        return graph


class NodeModel(nn.Module):
    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.N_nodes= None
        self.node_mlp = make_mlp(output_size)

    def forward(self, graph):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_features: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        N = graph['node_features'].size(1)

        feature= [graph['node_features']]
        for edge_set in graph['edge_sets']:
            feature.append(batch_scatter(edge_set['features'], edge_set['receivers'], reduction='mean', dim_size= N))

        feature = torch.cat(feature, dim=-1)

        feature= self.node_mlp(feature)
        graph['node_features']+= feature
        # print(graph['node_features'])
        return graph


class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model


    def forward(self, graph):
        if self.edge_model is not None:
            graph = self.edge_model(graph)
        if self.node_model is not None:
            graph = self.node_model(graph)

        if self.global_model is not None:
            graph = self.global_model(graph)

        return graph
 