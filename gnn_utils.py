import math
from collections import defaultdict
from urllib.parse import urlparse

import torch
from rdflib.compare import to_canonical_graph
from torch import Generator
from torch_geometric.data import HeteroData, Data
import rdflib
from torch.utils.data import DataLoader
from torch_geometric.nn import GAE
import numpy as np
import random

from src.gnn import GNN



def read_model_and_data(filename, to_evaluate, round=1, same_as_fraction_used_training=1, test_ratio=0.2, val_ratio=0):
    gae = torch.load(f'../models/gnn/{filename}.pt')
    (data, train_pos_edge_adj, test_pos_edge_adj, val_pos_edge_adj, node_from_ontology, nodes_ontology_1, nodes_ontology_2,
    nodes_dict, relations_dict) = read_data(filename, to_evaluate, round, same_as_fraction_used_training, test_ratio,
                                           val_ratio)

    return (gae, data, train_pos_edge_adj, test_pos_edge_adj, val_pos_edge_adj, node_from_ontology, nodes_ontology_1,
            nodes_ontology_2, nodes_dict, relations_dict)


def read_data(filename, to_evaluate, round=1, same_as_fraction_used_training=1, test_ratio=0.2, val_ratio=0, seed=0):

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    g = rdflib.Graph()
    g.parse(f'../data/round_{round}/{filename}.ttl', format='ttl')

    print(f'Triplets found: %d' % len(g))

    # find all relations and nodes
    relations = list(set(g.predicates()))
    nodes = list(set(g.subjects()).union(set(g.objects())))

    # Make sure it is deterministic
    relations = sorted(relations)
    nodes = sorted(nodes)

    nodes_dict, relations_dict = get_dictionaries(nodes, relations)

    ontology_namespace = get_unique_net_locs(nodes)

    nodes_ontology_1, nodes_ontology_2, node_from_ontology = get_nodes_per_ontology(nodes, nodes_dict, ontology_namespace)

    data = get_data(g, nodes_dict, relations_dict, node_from_ontology)
    data = split_edges(data, relations_dict, to_evaluate, same_as_fraction_used_training, test_ratio=test_ratio, val_ratio=val_ratio)

    train_pos_edge_adj = get_adj_dict(data.train_pos_edge_index)
    test_pos_edge_adj = get_adj_dict(data.test_pos_edge_index)
    val_pos_edge_adj = get_adj_dict(data.val_pos_edge_index) if val_ratio > 0 else None

    merged_pos_adj_test = defaultdict(list)
    for d in (train_pos_edge_adj, test_pos_edge_adj):
        for key, value in d.items():
            merged_pos_adj_test[key].extend(value)

    test_pos_edge_adj = dict(merged_pos_adj_test)

    if val_ratio > 0:
        merged_pos_adj_val = defaultdict(list)
        for d in (test_pos_edge_adj, val_pos_edge_adj):
            for key, value in d.items():
                merged_pos_adj_val[key].extend(value)

        val_pos_edge_adj = dict(merged_pos_adj_val)


    return (data, train_pos_edge_adj, test_pos_edge_adj, val_pos_edge_adj, node_from_ontology,
            nodes_ontology_1, nodes_ontology_2, nodes_dict, relations_dict)


def get_data(g, nodes_dict, relations_dict, node_from_ontology):
    edge_data = defaultdict(list)
    # Sort to ensure reproducibility
    sorted_triples = sorted(g.triples((None, None, None)))
    for s, p, o in sorted_triples:
        if s in nodes_dict and o in nodes_dict:
            src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
            # Just a check to be a 100% sure no other edges are added
            if node_from_ontology[src] != -1 and node_from_ontology[dst] != -1:
                edge_data['edge_index'].append([src, dst])
                edge_data['edge_type'].append(rel)
            else:
                raise Exception(f"Node not found in the ontology: {s} or {o}")

    data = Data(edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
                edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long))

    return data


def split_edges(data, relation_dict, to_evaluate, same_as_fraction_used_training, test_ratio=0.2, val_ratio=0):
    # First we check the input of to_evaluate and make sure its valid
    if type(to_evaluate) is str:
        # If the input is 'all' we evaluate all relations
        if to_evaluate == 'all':
            to_evaluate = relation_dict.keys()
        else:
            raise Exception(f"{to_evaluate} is not a valid string. Use 'all' or a list of relations.")
    elif isinstance(to_evaluate, rdflib.term.URIRef):
        raise Exception("to_evaluate must be a list of relations or 'all'.")

    row, col = data.edge_index
    edge_type = data.edge_type

    # These lists are used to store the masks for the edges of the relations we want to evaluate
    # For example: if we want to evaluate on sameAs, the mask of the entry corresponding to sameAs will be True if the
    # edge is of type sameAs and False otherwise. This is used to evaluate the model on the different relations,
    # separately and independently of each other.
    if val_ratio != 0: val_edge_eval_masks_list = []
    test_edge_eval_masks_list = []

    # We loop through all the relations that we want to evaluate
    for i, relation in enumerate(to_evaluate):
        # We get the mask which indicates if an edge is of the current relation to evaluate
        relation_mask = edge_type == relation_dict[relation]

        # We get the row, col and edge_type of the edges of the current relation
        relation_row = row[relation_mask]
        relation_col = col[relation_mask]
        relation_edge_type = edge_type[relation_mask]

        # We calculate the validation set size
        n_v = int(math.floor(val_ratio * relation_row.size(0)))
        # We calculate the test set size
        n_t = int(math.floor(test_ratio * relation_row.size(0)))

        # We shuffle the edges along with their types
        perm = torch.randperm(relation_row.size(0))
        relation_row, relation_col, relation_edge_type, = relation_row[perm], relation_col[perm], relation_edge_type[perm],

        # Validation set creation
        if n_v != 0:
            # We get the edges for the validation set of the current relation
            r, c, e = relation_row[:n_v], relation_col[:n_v], relation_edge_type[:n_v]

            # We check if the validation set already exists
            if 'val_pos_edge_index' in data:
                # If it exists we concatenate the new edges to the existing ones
                data.val_pos_edge_index = torch.cat([data.val_pos_edge_index, torch.stack([r, c], dim=0)], dim=1)
                data.val_edge_type = torch.cat([data.val_edge_type, e])
            else:
                # If it does not exist we create the validation set
                data.val_pos_edge_index = torch.stack([r, c], dim=0)
                data.val_edge_type = e
            val_edge_eval_masks_list.append(e == relation_dict[relation])

        # Test set creation
        # We get the edges for the test set of the current relation
        r, c, e = relation_row[n_v:n_v + n_t], relation_col[n_v:n_v + n_t], relation_edge_type[n_v:n_v + n_t]
        # We check if the test set already exists
        if 'test_pos_edge_index' in data:
            # If it exists we concatenate the new edges to the existing ones
            data.test_pos_edge_index = torch.cat([data.test_pos_edge_index, torch.stack([r, c], dim=0)], dim=1)
            data.test_edge_type = torch.cat([data.test_edge_type, e])
        else:
            # If it does not exist we create the test set
            data.test_pos_edge_index = torch.stack([r, c], dim=0)
            data.test_edge_type = e
        test_edge_eval_masks_list.append(e == relation_dict[relation])

        # Train set creation
        # We get the edges for the train set of the current relation
        r, c, e = relation_row[n_v + n_t:], relation_col[n_v + n_t:], relation_edge_type[n_v + n_t:]
        if relation == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#sameAs'):
            # We create a random permutation of the indices of the edges to be used in the train set
            indices = torch.randperm(len(r))[:int(len(r)*same_as_fraction_used_training)]
            r, c, e = r[indices], c[indices], e[indices]

        # We check if the train set already exists
        if 'train_pos_edge_index' in data:
            # If it exists we concatenate the new edges to the existing ones
            data.train_pos_edge_index = torch.cat([data.train_pos_edge_index, torch.stack([r, c], dim=0)], dim=1)
            data.train_edge_type = torch.cat([data.train_edge_type, e])
        else:
            # If it does not exist we create the train set
            data.train_pos_edge_index = torch.stack([r, c], dim=0)
            data.train_edge_type = e

    # We convert the list of masks to a single tensor. Each row will correspond to the mask of a relation.
    # The tensor is created only now as the size of the number of columns in the tensor is not known beforehand.
    if val_ratio != 0: data.val_edge_eval_masks = __tensor_list_to_tensor(val_edge_eval_masks_list, to_evaluate)
    data.test_edge_eval_masks = __tensor_list_to_tensor(test_edge_eval_masks_list, to_evaluate)

    # The train set includes the 1-test-ratio-val-ratio edges from the relations we want to evaluate.
    # The relations that we do not want to evaluate are all completely added to the train set.
    relations_not_evaluated = [relation for relation in relation_dict.keys() if relation not in to_evaluate]
    for relation in relations_not_evaluated:
        relation_mask = edge_type == relation_dict[relation]

        relation_row = row[relation_mask]
        relation_col = col[relation_mask]
        relation_edge_type = edge_type[relation_mask]

        # As there is always at least one relation we want to evaluate, we can safely concatenate the new edges to the existing ones
        data.train_pos_edge_index = torch.cat([data.train_pos_edge_index, torch.stack([relation_row, relation_col], dim=0)], dim=1)
        data.train_edge_type = torch.cat([data.train_edge_type, relation_edge_type])

    # We can create the masks for the edges that we want to evaluate in the train set based on the relations we want to evaluate
    data.train_edge_eval_masks = torch.zeros((len(to_evaluate), data.train_edge_type.size(0)), dtype=torch.bool)
    for i, relation in enumerate(to_evaluate):
        data.train_edge_eval_masks[i] = data.train_edge_type == relation_dict[relation]

    return data



def __tensor_list_to_tensor(tensor_list, to_evaluate):
    """
        Converts a list of tensors into a single tensor with each original tensor
        placed in specific positions within a larger zero-padded tensor.

        Args:
            tensor_list (list of torch.Tensor): A list of 1D tensors to be combined.
            to_evaluate (list): A list indicating the number of rows in the resulting tensor.

        Returns:
            torch.Tensor: A 2D tensor where each row corresponds to a tensor from
            the list, placed at specific positions, and padded with zeros.

        Example:
            tensor_list = [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1, 1])]
            to_evaluate = [relation1, relation2, relation3]

            Will give:
            tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    """
    total_length = sum([mask.size(0) for mask in tensor_list])
    resulting_tensor = torch.zeros((len(to_evaluate), total_length), dtype=torch.bool)

    current_position = 0
    for i, tensor in enumerate(tensor_list):
        # Get the length of the current tensor
        tensor_length = tensor.size(0)
        # Copy the current tensor to the result_tensor at the current position
        resulting_tensor[i, current_position:current_position + tensor_length] = tensor
        # Update the current position
        current_position += tensor_length

    return resulting_tensor


def get_unique_net_locs(nodes):
    unique_net_locs = set()
    for node in nodes:
        if not isinstance(node, rdflib.term.BNode):
            netloc = get_net_loc_from_node(node)
            unique_net_locs.add(netloc)
            if len(unique_net_locs) == 2:
                break

    net_locs_dict = {netloc: index for index, netloc in enumerate(unique_net_locs)}
    return net_locs_dict


def get_net_loc_from_node(node):
    netloc = urlparse(str(node)).netloc
    # We remove the "www." starting string of the netloc, as not all nodes have it even though they are from the same ontology
    if netloc.startswith("www."):
        netloc = netloc.replace("www.", "")
    return netloc


def get_dictionaries(nodes, relations):
    nodes_without_thing = [node for node in nodes if node != rdflib.term.URIRef("http://www.w3.org/2002/07/owl#Thing")]
    nodes_without_BNodes = [node for node in nodes_without_thing if not isinstance(node, rdflib.term.BNode)]

    relations_dict = {rel: i for i, rel in enumerate(relations)}
    nodes_dict = {node: i for i, node in enumerate(nodes_without_BNodes)}

    return nodes_dict, relations_dict


def get_nodes_per_ontology(nodes, nodes_dict, ontology_namespace):
    # Dictionary as not all indices are guaranteed to be present, probably due to the "Thing" node
    node_from_ontology_dict = {}

    for node in nodes:
        namespace = get_net_loc_from_node(node)
        if namespace in ontology_namespace:
            node_from_ontology_dict[nodes_dict[node]] = ontology_namespace[namespace]

    nodes_ontology_1 = torch.tensor([node for node, ontology in node_from_ontology_dict.items() if ontology == 0])
    nodes_ontology_2 = torch.tensor([node for node, ontology in node_from_ontology_dict.items() if ontology == 1])

    switched_dict = {v: k for k, v in nodes_dict.items()}
    for key, value in node_from_ontology_dict.items():
        if key in switched_dict:
            if switched_dict[key] == 'http://www.omim.org/phenotypicSeries/PS208540':
                print(switched_dict[key])

    node_from_ontology = [-1] * (max(node_from_ontology_dict.keys()) + 1)

    for key, value in node_from_ontology_dict.items():
        node_from_ontology[key] = value

    return nodes_ontology_1, nodes_ontology_2, node_from_ontology


def get_edges_per_ontology(edge_index, node_from_ontology):
    edges_in_same_ontology = []
    edges_between_ontologies = []
    for i in range(edge_index.size(1)):
        print(node_from_ontology[edge_index[0, i].item()], node_from_ontology[edge_index[1, i].item()])
        if node_from_ontology[edge_index[0, i].item()] != node_from_ontology[edge_index[1, i].item()]:
            edges_between_ontologies.append([edge_index[0, i], edge_index[1, i]])
        else:
            edges_in_same_ontology.append([edge_index[0, i], edge_index[1, i]])

    return (torch.tensor(edges_in_same_ontology, dtype=torch.long).t().contiguous(),
            torch.tensor(edges_between_ontologies, dtype=torch.long).t().contiguous())


def get_adj_dict(edge_index):
    adjacency_list = defaultdict(list)

    # Iterate over the edges
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()

        adjacency_list[src].append(dst)

    return dict(adjacency_list)


def get_dataloaders(data, to_evaluate, batch_size_train, batch_size_test, batch_size_val=None):
    class HeteroDataset(torch.utils.data.Dataset):
        def __init__(self, data, split):
            self.data = data
            self.split = split

        def __len__(self):
            return self.data[self.split + '_pos_edge_index'].size(1)

        def __getitem__(self, idx):
            edge_index = self.data[self.split + '_pos_edge_index'][:, idx].view(2, 1)
            edge_type = self.data[self.split + '_edge_type'][idx].view(1)
            edge_eval_masks = self.data[self.split + '_edge_eval_masks'][:, idx].view(len(to_evaluate), 1)

            return HeteroData(edge_index=edge_index, edge_type=edge_type, edge_eval_masks=edge_eval_masks)

    if batch_size_train == 'all':
        batch_size_train = data.train_pos_edge_index.size(1)
    if batch_size_test == 'all':
        batch_size_test = data.test_pos_edge_index.size(1)
    if batch_size_val is not None and batch_size_val == 'all':
        batch_size_val = data.val_pos_edge_index.size(1)

    train_dataset = HeteroDataset(data, 'train')
    val_dataset = HeteroDataset(data, 'val') if 'val_pos_edge_index' in data else None
    test_dataset = HeteroDataset(data, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=__collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=__collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=__collate_fn)

    return train_loader, test_loader, val_loader


# Custom collate function to handle batching of HeteroData
def __collate_fn(batch):
    edge_index_list = []
    edge_type_list = []
    edge_eval_masks_list = []
    for data in batch:
        edge_index_list.append(data.edge_index)
        edge_type_list.append(data.edge_type)
        edge_eval_masks_list.append(data.edge_eval_masks)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_type = torch.cat(edge_type_list, dim=0)
    edge_eval_masks = torch.cat(edge_eval_masks_list, dim=1)

    batched_data = HeteroData()
    batched_data.edge_index = edge_index
    batched_data.edge_type = edge_type
    batched_data.edge_eval_masks = edge_eval_masks

    return batched_data


def copy_graph(g):
    new_g = rdflib.Graph()

    for triple in g:
        new_g.add(triple)

    return new_g
