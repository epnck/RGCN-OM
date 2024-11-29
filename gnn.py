import numpy as np
import rdflib
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn import Parameter
from torch_geometric.nn import GAE, RGCNConv
from torch_geometric.utils.negative_sampling import edge_index_to_vector, vector_to_edge_index, sample
from tqdm import tqdm


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations, num_layers, dropout_prob):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            RGCNConv(hidden_channels, hidden_channels, num_relations, num_blocks=5)
            for _ in range(num_layers)
        ])
        self.dropout_prob = dropout_prob
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != self.num_layers - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


class GNN():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # MPS doesn't work with the torch.isin() function yet
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    def __init__(self, num_nodes, num_relations, same_as_index, relation_ratio, node_from_ontology, nodes_ontology_1,
                 nodes_ontology_2, model=None, num_layers=3, hidden_channels=200, learning_rate=5e-3, dropout_prob=0.2,
                 regularization_val=1e-2, seed=10):
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.same_as_index = same_as_index
        self.regularization_val = regularization_val
        self.node_from_ontology = torch.tensor(node_from_ontology, device='cpu')
        self.nodes_ontology_1 = nodes_ontology_1.to('cpu')
        self.nodes_ontology_2 = nodes_ontology_2.to('cpu')
        self.relation_ratio = relation_ratio.to(self.device)
        if model is None:
            self.model = GAE(RGCNEncoder(num_nodes, hidden_channels, num_relations, num_layers, dropout_prob),
                             DistMultDecoder(num_relations, hidden_channels))
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model = self.model.to(self.device)

        torch.manual_seed(seed)

    def train(self, data_loader, all_train_indices):
        self.model.train()
        epoch_loss = 0

        for data in tqdm(data_loader, desc="training"):
            data.edge_index = data.edge_index.to(self.device)
            data.edge_type = data.edge_type.to(self.device)

            self.optimizer.zero_grad()

            z = self.model.encode(data.edge_index, data.edge_type)

            pos_out = self.model.decode(z, data.edge_index, data.edge_type)

            neg_edge_index = torch.tensor([], dtype=torch.long, device=self.device)
            neg_edge_types = torch.tensor([], dtype=torch.long, device=self.device)
            for i in range(self.num_relations):
                if i in data.edge_type:
                    neg_edge_index_relation = sample_negative_edges(edge_index=all_train_indices, # All train indices
                        # to make sure that we don't sample negative edges which do exist in a different batch
                                                                    nodes_ontology_1=self.nodes_ontology_1,
                                                                    nodes_ontology_2=self.nodes_ontology_2,
                                                                    sampling_same_as_edges=i == self.same_as_index,
                                                                    num_nodes=self.num_nodes,
                                                                    num_neg_samples=int(data.edge_index.size(1)*self.relation_ratio[i])).to(self.device)
                    neg_edge_index = torch.cat([neg_edge_index, neg_edge_index_relation], dim=1)
                    neg_edge_types = torch.cat([neg_edge_types, torch.full((neg_edge_index_relation.size(1),), i, dtype=torch.long, device=self.device)], dim=0)

            neg_out = self.model.decode(z, neg_edge_index, neg_edge_types)

            out = torch.cat([pos_out, neg_out])
            ground_truth = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
            perm = torch.randperm(out.size(0))

            # Shuffle 'out' and 'ground_truth' tensors using the permutation
            shuffled_out = out[perm]
            shuffled_ground_truth = ground_truth[perm]

            cross_entropy_loss = F.binary_cross_entropy_with_logits(shuffled_out, shuffled_ground_truth)
            reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
            loss = cross_entropy_loss + self.regularization_val * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    @torch.no_grad()
    def eval(self, data_loader, edge_adj, evaluated_relations, num_neg_edges=100):
        self.model.eval()
        hits1_total = torch.zeros(len(evaluated_relations), dtype=torch.float64, device=self.device)
        hits_10_total = torch.zeros(len(evaluated_relations), dtype=torch.float64, device=self.device)
        mr_total = torch.zeros(len(evaluated_relations), dtype=torch.float64, device=self.device)
        mrr_total = torch.zeros(len(evaluated_relations), dtype=torch.float64, device=self.device)

        # Use a counter to keep track of the number of batches where the corresponding relation occurred in
        counter = torch.zeros(len(evaluated_relations), dtype=torch.int64, device=self.device)
        for data in tqdm(data_loader, desc="evaluating"):
            data.edge_index = data.edge_index.to(self.device)
            data.edge_type = data.edge_type.to(self.device)
            data.edge_eval_masks = data.edge_eval_masks.to(self.device)
            # We want to evaluate all relations separately, so we loop over the entries in the edge_eval_masks
            # and evaluate each mask separately
            for i, edge_eval_mask in enumerate(data.edge_eval_masks):
                num_eval_edges = edge_eval_mask.sum()
                # Check if there are any edges to evaluate
                if num_eval_edges > 0:
                    output = self.model.encode(data.edge_index, data.edge_type)

                    hits1, hits10, mr, mrr = self._eval_hits(edge_index=data.edge_index,
                                                             edge_type=data.edge_type,
                                                             edge_eval_mask=edge_eval_mask,
                                                             edge_adj=edge_adj,
                                                             tail_pred=1,
                                                             output=output,
                                                             num_eval_edges=num_eval_edges,
                                                             max_num=num_neg_edges,
                                                             relation=evaluated_relations[i])

                    hits1_total[i] += hits1
                    hits_10_total[i] += hits10
                    mr_total[i] += mr
                    mrr_total[i] += mrr
                    counter[i] += 1

        # Return the averages
        return hits1_total / counter, hits_10_total / counter, mr_total / counter, mrr_total / counter

    def to(self, device):
        self.model.to(device)
        return self


    @torch.no_grad()
    def eval_PRF(self, data_loader, rel_num, negative_ratio=1):
        self.model.eval()
        labels = torch.empty((0), dtype=torch.bool, device=self.device)
        predicted = torch.empty((0), dtype=torch.bool, device=self.device)
        for data in data_loader:
            edge_index = data.edge_index.to(self.device)
            edge_type = data.edge_type.to(self.device)
            edge_eval_mask = data.edge_eval_masks[rel_num].to(self.device)

            masked_edge_index = torch.stack([edge_index[0][edge_eval_mask], edge_index[1][edge_eval_mask]], dim=0)
            masked_edge_type = edge_type[edge_eval_mask]  # This should always be the same number in this array
            # Check if there are any edges to evaluate
            if edge_eval_mask.sum() > 0:
                if len(masked_edge_type.unique()) != 1:
                    raise Exception(
                        '''An error occurred: There are multiple encodings for the same as relationship array. This should never
                        happen. There could be either zero sameAs relations in the graph or multiple encodings.
                        The number of encodings for the SameAs node given is {}'''.format(len(masked_edge_type.unique()))
                    )

                # For the real part
                node_embeddings = self.model.encode(masked_edge_index, masked_edge_type)
                scores_real = self.model.decode(node_embeddings, masked_edge_index, masked_edge_type)
                scores_real_probability = torch.sigmoid(scores_real)

                labels = torch.cat([labels, torch.ones(len(scores_real_probability), dtype=torch.bool, device=self.device)])
                predicted = torch.cat([predicted, torch.round(scores_real_probability).type(torch.bool)])

                # increasing negative_ratio will make more negative edges than positive edges, one will balance them
                n = round(len(masked_edge_type) * negative_ratio)

                negative_edge_index = sample_negative_edges(edge_index=masked_edge_index.to('cpu'),
                                                                nodes_ontology_1=self.nodes_ontology_1,
                                                                nodes_ontology_2=self.nodes_ontology_2,
                                                                sampling_same_as_edges=rel_num == self.same_as_index,
                                                                num_nodes=self.num_nodes,
                                                                num_neg_samples=n).to(self.device)


                negative_edge_type = torch.full((negative_edge_index.size(1),), rel_num, dtype=torch.long, device=self.device)

                # Prediction for the negative relations
                #node_embeddings_negative = self.model.encode(negative_edge_index,
                #                                              negative_edge_type)
                scores_negative = self.model.decode(node_embeddings, negative_edge_index,
                                                     negative_edge_type)
                scores_negative_probability = torch.sigmoid(scores_negative)

                labels = torch.cat([labels, torch.zeros(len(scores_negative_probability), dtype=torch.bool, device=self.device)])
                predicted = torch.cat([predicted, torch.round(scores_negative_probability).type(torch.bool)])

        labels = labels.to('cpu')
        predicted = predicted.to('cpu')

        precision_pos = precision_score(labels, predicted, labels=[0, 1], pos_label=1)
        recall_pos = recall_score(labels, predicted, labels=[0, 1], pos_label=1)
        f1_pos = f1_score(labels, predicted, labels=[0, 1], pos_label=1)

        precision_neg = precision_score(labels, predicted, labels=[0, 1], pos_label=0)
        recall_neg = recall_score(labels, predicted, labels=[0, 1], pos_label=0)
        f1_neg = f1_score(labels, predicted, labels=[0, 1], pos_label=0)

        precision_avg = 1/2 * (precision_pos + precision_neg)
        recall_avg = 1/2 * (recall_pos + recall_neg)
        f1_avg = 1/2 * (f1_pos + f1_neg)

        return (precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg,
                precision_avg, recall_avg, f1_avg)

    @torch.no_grad()
    def _eval_hits(self, edge_index, edge_type, edge_eval_mask, edge_adj, tail_pred, output, num_eval_edges, max_num, relation):
        # Get all node embeddings for either head or tail nodes based on tail_pred
        # 1-tail_pred is 0 if tail_pred is 1 and vice versa. I.e., it gets the node embedding of the real head or tail.
        # We only get the embeddings of the nodes that are part of the edges we want to evaluate
        x = torch.index_select(output, 0, edge_index[1 - tail_pred][edge_eval_mask])

        # The edge_type[edge_eval_index] is a tensor which should only contain the type of the edges we want to evaluate
        # For example, in the case of only evaluating the sameAs type for human2mouse, this tensor will only contain
        # zeros as the dictionary maps the sameAs relation to 0.
        rel_emb = torch.index_select(self.model.decoder.rel_emb, 0, edge_type[edge_eval_mask])

        # Sample negative edges indices
        candidates, candidates_embeds = sample_negative_edges_head_or_tail(edge_index=edge_index,
                                                                           edge_adj=edge_adj,
                                                                           tail_pred=tail_pred,
                                                                           output=output,
                                                                           edge_eval_mask=edge_eval_mask,
                                                                           node_from_ontology=self.node_from_ontology,
                                                                           nodes_ontology_1=self.nodes_ontology_1,
                                                                           nodes_ontology_2=self.nodes_ontology_2,
                                                                           num_eval_edges=num_eval_edges,
                                                                           max_num=max_num,
                                                                           relation=relation)

        # Simulate the decoder output. Should probably change the decoder such that we can input multiple instances,
        # then we don't have to simulate the decoder here instead.
        decoder_output = torch.sum(x.unsqueeze(1) * rel_emb.unsqueeze(1) * candidates_embeds, dim=2)

        # Take the sigmoid of the decoder output to get probabilities
        probabilities = torch.sigmoid(decoder_output)

        # For distances, a lower value is better, so we sort in ascending order
        # For probabilities, a higher probability is better, so we sort in descending order
        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        true_edge_ranks = torch.zeros(num_eval_edges, dtype=torch.int64)

        # Find the rank of the true edge within the sorted distances
        for i in range(num_eval_edges):
            # The true edge is always the last candidate, which has index max_num - 1
            true_edge_rank = (sorted_indices[i] == max_num - 1).nonzero(as_tuple=True)[0]
            true_edge_ranks[i] = true_edge_rank

        # Calculate top1 and top10
        top1 = (true_edge_ranks < 1).sum().item()
        top10 = (true_edge_ranks < 10).sum().item()
        mr = true_edge_ranks.float().mean().item()
        mrr = 1 / mr if mr > 0 else 1

        return top1 / num_eval_edges, top10 / num_eval_edges, mr, mrr


# *** HELPER FUNCTIONS *** #

# Adjustment of the PyTorchs' negative sampling function to allow for sampling of negative edges based on the ontology
def sample_negative_edges(edge_index, nodes_ontology_1, nodes_ontology_2, sampling_same_as_edges, num_nodes, num_neg_samples):
    size = (num_nodes, num_nodes)
    idx, population = edge_index_to_vector(edge_index, size, False, False)

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    # Just in case if the sample size is 0, we set it to 1 to avoid errors
    if sample_size == 0:
        sample_size = 1

    if not sampling_same_as_edges:
        ratio = len(nodes_ontology_1) / (len(nodes_ontology_1) + len(nodes_ontology_2))
        sample_size *= 2
        sample_size_ontology_1 = int(ratio * sample_size)
        sample_size_ontology_2 = sample_size - sample_size_ontology_1

    idx = idx.to('cpu')
    neg_idx = None

    for _ in range(3):
        if sampling_same_as_edges:
            rnd_1 = np.random.randint(0, len(nodes_ontology_1), sample_size)
            rnd_2 = np.random.randint(0, len(nodes_ontology_2), sample_size)

            sampled_edges = torch.stack([nodes_ontology_1[rnd_1], nodes_ontology_2[rnd_2]], dim=0)

            # Below we swap the direction of the edges with a 50% probability, such that we do not always get edges
            # from ontology 1 to ontology 2, but also from ontology 2 to ontology 1
            mask = torch.randperm(sampled_edges.size(1)) % 2

            # Create a copy of the original tensor to perform swapping
            shuffled_edges = sampled_edges.clone()

            # Swap the values based on the mask
            shuffled_edges[0, mask.bool()] = sampled_edges[1, mask.bool()]
            shuffled_edges[1, mask.bool()] = sampled_edges[0, mask.bool()]

            rnd, _ = edge_index_to_vector(sampled_edges, size, False, False)
        else:
            # Sample a random number of edge indices from all edges that could exist
            if sample_size_ontology_1 > 0:
                rnd_1 = np.random.randint(0, len(nodes_ontology_1), sample_size_ontology_1)
                half = int(len(rnd_1) / 2)
                sampled_edges_1 = torch.stack([nodes_ontology_1[rnd_1[:half]], nodes_ontology_1[rnd_1[half:2*half]]], dim=0)
            else:
                sampled_edges_1 = torch.empty((2, 0), dtype=torch.long)

            if sample_size_ontology_2 > 0:
                rnd_2 =  np.random.randint(0, len(nodes_ontology_2), sample_size_ontology_2)
                half = int(len(rnd_2) / 2)
                sampled_edges_2 = torch.stack([nodes_ontology_2[rnd_2[:half]], nodes_ontology_2[rnd_2[half:2*half]]], dim=0)
            else:
                sampled_edges_2 = torch.empty((2, 0), dtype=torch.long)

            sampled_edges = torch.cat([sampled_edges_1, sampled_edges_2], dim=1)
            rnd, _ = edge_index_to_vector(sampled_edges, size, False, False)

        mask = np.isin(rnd, idx)

        if neg_idx is not None:
            mask |= np.isin(rnd, neg_idx.to('cpu'))
        mask = torch.from_numpy(mask).to(torch.bool)

        # We select the indices that are valid, i.e. the sampled edge does not exist in the graph or in the list of the negative indices
        rnd = rnd[~mask]
        neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])

        # We sampled more negative indices than required, so we only keep the required number of negative indices
        if neg_idx.numel() >= num_neg_samples:
            neg_idx = neg_idx[:num_neg_samples]
            break

    return vector_to_edge_index(neg_idx, size, False, False)


def sample_negative_edges_head_or_tail(edge_index, edge_adj, tail_pred, output, edge_eval_mask, node_from_ontology, nodes_ontology_1, nodes_ontology_2, num_eval_edges, max_num, relation):
    # Prepare the tensor for negative tail/head candidates
    candidates = torch.zeros((num_eval_edges, max_num), dtype=torch.long, device=edge_index.device)

    masked_edge_index = torch.stack([edge_index[0][edge_eval_mask], edge_index[1][edge_eval_mask]], dim=0)

    # Vectorized way to sample negative tails
    index = 0
    for i in range(masked_edge_index.size(1)):
        # Check if the relation is sameAs, as this relation needs special treatment
        if relation == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#sameAs'):
            # Head prediction
            if tail_pred == 0:
                # If the head is in the first ontology, we sample from the first ontology and vice versa
                nodes = nodes_ontology_1 if node_from_ontology[masked_edge_index[0][i].item()] == 0 else nodes_ontology_2

            # Tail prediction
            elif tail_pred == 1:
                # If the tail is in the first ontology, we sample from the first ontology and vice versa
                nodes = nodes_ontology_1 if node_from_ontology[masked_edge_index[1][i].item()] == 0 else nodes_ontology_2
        else:
            # As we are not evaluating the sameAs relations, the two endpoints of an edge are always in the same
            # ontology. Thus, we can sample from the same ontology as the head/tail.
            nodes = nodes_ontology_1 if node_from_ontology[masked_edge_index[0][i].item()] == 0 else nodes_ontology_2

        # Exclude all nodes (tails) that are already connected to the head, for tail prediction
        # For head prediction we exclude all heads that are connected to the tail
        exclude_tensor = torch.tensor(edge_adj[masked_edge_index[1 - tail_pred, i].item()])
        mask = ~torch.isin(nodes, exclude_tensor)  # Does not work with an MPS device
        remaining_nodes = nodes[mask]

        # Randomly sample max_num nodes from the remaining nodes
        sampled_indices = torch.randperm(remaining_nodes.size(0))[:max_num]
        sampled_tails = remaining_nodes[sampled_indices]
        sampled_tails[-1] = masked_edge_index[tail_pred, i].item()  # Add the true tail as the last element
        candidates[index] = sampled_tails

        index += 1


    # Get the node embeddings for the negative tail/head candidates
    candidates_embeds = torch.index_select(output, 0, candidates.view(-1)).view(num_eval_edges, max_num, -1)

    return candidates, candidates_embeds
