# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from collections import Counter

from . import data_utils, FairseqDataset

logger = logging.getLogger(__name__)


def cons_batch_graph(graphs, old_padding_idx, new_padding_idx=1):
    g_ids = {}
    g_edge_features = {}
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_fw_edge = {}
    g_bw_edge = {}
    g_nodes = []
    g_edges = []
    n_graphs = graphs['nodes'].shape[0]
    nodes, edges, in_indices, in_edges, out_indices, out_edges = graphs.values()

    def cons(data, gid_map, var):
        for id, adj in enumerate(data):
            g_id = gid_map[id]
            if g_id not in var:
                var[g_id] = []
            for t in adj:
                if t == old_padding_idx:
                    g_t = new_padding_idx
                else:
                    g_t = id_gid_map[t]
                var[g_id].append(g_t)
        # return result

    for i in range(n_graphs):
        ids = nodes.shape[1]
        in_indices_ = in_indices[i].numpy().tolist()
        out_indices_ = out_indices[i].numpy().tolist()
        in_edges_ = in_edges[i].numpy().tolist()
        out_edges_ = out_edges[i].numpy().tolist()
        features = nodes[i].numpy().tolist()
        edges_feat = edges[i].numpy().tolist()
        nodes_ = []
        edges_ = []
        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in range(ids):
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[id]
            g_edge_features[offset + id] = edges_feat[id]
            id_gid_map[id] = offset + id
            nodes_.append(offset + id)
            edges_.append(offset + id)
        g_nodes.append(nodes_)
        g_edges.append(edges_)
        cons(in_indices_, id_gid_map, g_bw_adj)
        cons(out_indices_, id_gid_map, g_fw_adj)
        cons(out_edges_, id_gid_map, g_fw_edge)
        cons(in_edges_, id_gid_map, g_bw_edge)
    node_size = len(g_ids.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []
    g_ids_features[len(g_ids_features)] = 1
    g_edge_features[len(g_edge_features)] = 1
    g_fw_adj[len(g_fw_adj)] = [node_size for _ in range(len(g_fw_adj[0]))]
    g_bw_adj[len(g_bw_adj)] = [node_size for _ in range(len(g_bw_adj[0]))]
    g_fw_edge[len(g_bw_edge)] = [node_size for _ in range(len(g_fw_edge[0]))]
    g_bw_edge[len(g_bw_edge)] = [node_size for _ in range(len(g_bw_edge[0]))]

    graph = {}
    graph['g_ids'] = g_ids
    graph['nodes_features'] = torch.from_numpy(np.array(list(g_ids_features.values())))
    graph['edges_features'] = torch.from_numpy(np.array(list(g_edge_features.values())))
    graph['nodes'] = torch.from_numpy(np.array(g_nodes))
    graph['edges'] = torch.from_numpy(np.array(g_edges))
    graph['out_indices'] = torch.from_numpy(np.array(list(g_fw_adj.values()), dtype=np.float))
    graph['out_edges'] = torch.from_numpy(np.array(list(g_fw_edge.values()), dtype=np.float))
    graph['in_indices'] = torch.from_numpy(np.array(list(g_bw_adj.values()), dtype=np.float))
    graph['in_edges'] = torch.from_numpy(np.array(list(g_bw_edge.values()), dtype=np.float))
    return graph


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, data, left_pad=None, move_eos_to_beginning=False, amr=False, array_3d=False):
        if amr is False:
            values = [s[key] for s in data]
        else:
            values = [s['amr'][key] for s in data]
        return data_utils.collate_tokens(
            values,
            pad_idx, eos_idx, left_pad, move_eos_to_beginning, array_3d
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', samples, left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    graph_tokens = None
    graph_lengths = None
    if samples[0]['amr'] is not None:
        nodes = merge('node', samples, amr=True)
        nodes = nodes.index_select(0, sort_order)
        edges = merge('edge', samples, amr=True)
        edges = edges.index_select(0, sort_order)
        amr = [sample['amr'] for sample in samples]
        assert len(amr) == len(sort_order)
        amr = [amr[sort_order[i]] for i in range(len(sort_order))]
        nnodes = torch.LongTensor([a['node'].numel() for a in amr])
        in_edges = []
        in_indices = []
        out_edges = []
        out_indices = []
        for i, graph in enumerate(amr):
            in_indices.append(graph['in_indices'].reshape(nnodes[i], -1).numpy().tolist())
            in_edges.append(graph['in_edges'].reshape(nnodes[i], -1).numpy().tolist())
            out_indices.append(graph['out_indices'].reshape(nnodes[i], -1).numpy().tolist())
            out_edges.append(graph['out_edges'].reshape(nnodes[i], -1).numpy().tolist())

        old_padding_value = 0
        for value_i in in_indices:
            for value_ij in value_i:
                tmp = max(value_ij)
                old_padding_value = tmp if tmp > old_padding_value else old_padding_value
        batch_size, node_num = nodes.shape
        new_padding_value = batch_size * node_num

        in_indices = data_utils.collate_tokens(in_indices, pad_idx, array_3d=True)
        out_indices = data_utils.collate_tokens(out_indices, pad_idx, array_3d=True)
        in_edges = data_utils.collate_tokens(in_edges, pad_idx, array_3d=True)
        out_edges = data_utils.collate_tokens(out_edges, pad_idx, array_3d=True)
        graph_tokens = {
            'nodes': nodes,
            'edges': edges,
            'in_indices': in_indices,
            'in_edges': in_edges,
            'out_indices': out_indices,
            'out_edges': out_edges}
        graph_lengths = {'nnodes': nnodes,
                         }
        graph_tokens = cons_batch_graph(graph_tokens, old_padding_value, new_padding_value)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', samples, left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target', samples,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'graph_tokens': graph_tokens,
            'graph_lengths': graph_lengths  # n_in_indices and n_out_indices are equal to nnodes
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
            self, src, src_sizes, src_dict,
            amr=None, amr_sizes=None, amr_dict=None,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            align_dataset=None,
            append_bos=False, eos=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.amr = amr
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.amr_size = np.array(amr_sizes) if amr is not None else None
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.amr_dict = amr_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        amr_item = None
        if self.amr is not None:
            # node_ids,edge_ids, in_neigh_edge_ids, in_neigh_indice_ids, out_neigh_edge_ids, out_neigh_indice_ids
            amr_idx = index * 6
            nodes = self.amr[amr_idx]
            edges = self.amr[amr_idx + 1]
            in_neigh_edges = self.amr[amr_idx + 2]
            in_neigh_indices = self.amr[amr_idx + 3]
            out_neigh_edges = self.amr[amr_idx + 4]
            out_neigh_indices = self.amr[amr_idx + 5]
            amr_item = {
                'node': nodes,
                'edge': edges,
                'in_indices': in_neigh_indices,
                'in_edges': in_neigh_edges,
                'out_indices': out_neigh_indices,
                'out_edges': out_neigh_edges
            }
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'amr': amr_item,
            'target': tgt_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        result = collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
        return result

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
