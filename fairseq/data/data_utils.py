# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import sys
import types

import numpy as np
from fairseq.data.amr_utils import read_anonymized

logger = logging.getLogger(__name__)


def infer_language_pair(path, with_amr=False):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    if with_amr:
        amr = None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            src, dst = parts[1].split("-")
            if with_amr:
                amr = parts[2] if parts[2] != src and parts[2] != dst else None
                if amr is not None:
                    return src, amr, dst
            else:
                return src, None, dst
    if with_amr:
        return src, None, dst
    return src, dst


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False, array_3d=False,
                   is_amr=False, old_padding_value=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    def copy_tensor(src, dst):
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src, dtype=torch.long)
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if array_3d:
        dim1 = len(values)
        dim2 = max([len(item) for item in values])
        dim3 = max([len(item) for sub in values for item in sub])
        if is_amr:
            pad_idx = dim1 * dim2
        res = torch.LongTensor(dim1, dim2, dim3).fill_(pad_idx)

        for i, value_i in enumerate(values):
            for j, value_ij in enumerate(value_i):
                counter = 0
                max_n_loops = len(value_ij)
                while old_padding_value in value_ij:
                    idx_ = value_ij.index(old_padding_value)
                    value_ij[idx_] = pad_idx
                    counter += 1
                    if counter == max_n_loops:
                        break
                copy_tensor(value_ij, res[i, j][:len(value_ij)])

    else:
        size = max(v.size(0) for v in values)
        # if is_amr:
        #     pad_idx = len(values) * size
        res = values[0].new(len(values), size).fill_(pad_idx)
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def read_amr(line, lower=True):
    max_in_neigh = 0
    max_out_neigh = 0
    max_node = 0
    max_sent = 0

    amr = line
    amr_node = []
    edges = []
    amr_edge = []
    read_anonymized(amr.strip().split(), amr_node, amr_edge)
    # 1.
    # nodes.append(amr_node)
    # 2. & 3.
    edges.append(":self")
    edges += [x[2] for x in amr_edge]
    in_indices = [[i, ] for i, x in enumerate(amr_node)]
    in_edges = [[0, ] for i, x in enumerate(amr_node)]
    out_indices = [[i, ] for i, x in enumerate(amr_node)]
    out_edges = [[0, ] for i, x in enumerate(amr_node)]
    for (i, j, lb) in amr_edge:
        in_indices[j].append(i)
        in_edges[j].append(edges.index(lb))
        out_indices[i].append(j)
        out_edges[i].append(edges.index(lb))
    max_in_neigh = max(max_in_neigh, max(len(x) for x in in_indices))
    max_out_neigh = max(max_out_neigh, max(len(x) for x in out_indices))
    max_node = max(max_node, len(amr_node))
    return amr_node, edges, in_indices, in_edges, out_indices, out_edges, \
           max_node, max_in_neigh, max_out_neigh, max_sent


def load_indexed_dataset(path, dictionary, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        if "amr" in path:
            logger.info('loaded {} examples from: {}'.format(int(len(dataset) / 6), path_k))
        else:
            logger.info('loaded {} examples from: {}'.format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # Hacky as heck, for the specific case of multilingual training with RoundRobin.
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):
                return all(
                    a is None or b is None or a <= b
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception((
                            'Size of sample #{} is invalid (={}) since max_positions={}, '
                            'skip this example with --skip-invalid-size-inputs-valid-test'
                        ).format(ignored[0], dataset.size(ignored[0]), max_positions))
    if len(ignored) > 0:
        logger.warning((
                           '{} samples have invalid sizes and will be skipped, '
                           'max_positions={}, first few sample ids={}'
                       ).format(len(ignored), max_positions, ignored[:10]))
    return indices


def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    try:
        from fairseq.data.data_utils_fast import batch_by_size_fast
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    return batch_by_size_fast(indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult)


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol == '_EOW':
        sentence = sentence.replace(' ', '').replace('_EOW', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence
