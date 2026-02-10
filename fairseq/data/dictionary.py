# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
import numpy as np


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            pad="<pad>",
            eos="</s>",
            unk="<unk>",
            bos="<s>",
            extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False, extra_symbols_to_ignore=None):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())
            sent = " ".join(
                token_string(i)
                for i in tensor if i.item() not in extra_symbols_to_ignore
            )
        else:
            sent = " ".join(token_string(i) for i in tensor if i.item() not in extra_symbols_to_ignore)

        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def pad4amr(self):
        return len(self)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with PathManager.open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                            .format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial:],
                ex_vals + self.count[self.nspecial:],
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
            self,
            line,
            line_tokenizer=tokenize_line,
            add_if_not_exist=True,
            consumer=None,
            append_eos=True,
            reverse_order=False,
    ):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    def encode_graph_info(self, graph,
                          max_node=60,
                          max_in_neighbor=2,
                          max_out_neighbor=10,
                          add_if_not_exist=False,
                          consumer=None,
                          append_eos=False,
                          padding=True
                          ):
        nodes, edges, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, _, max_in_neigh, max_out_neigh, max_sent = data_utils.read_amr(
            graph)
        nnodes = min(len(nodes), max_node)

        def padding_data(data, max_neigh=-1, is_indice=False):
            dim0 = len(data)
            dim1 = max([len(item) for item in data])
            dim1 = min(dim1, max_neigh)
            result = np.zeros((dim0, dim1), dtype=np.int32)
            result.fill(self.pad4amr())
            for i in range(dim0):
                cur_item = data[i]
                cur_size = min(len(cur_item), dim1)
                result[i, :cur_size] = cur_item[:cur_size]
            result = torch.from_numpy(result)
            # filter out indices with index = nnodes
            if is_indice is True:
                idxs = (result >= nnodes).nonzero()
                for idx in idxs:
                    result[idx[0], [idx[1]]] = self.pad4amr()
            return result

        def encode(info, length, append_eos, consumer):
            ids = torch.IntTensor(length + 1 if append_eos else length)
            for i, item in enumerate(info):
                if add_if_not_exist:
                    idx = self.add_symbol(item)
                else:
                    idx = self.index(item)
                if consumer is not None:
                    consumer(item, idx)
                ids[i] = idx
            if append_eos:
                ids[nnodes] = self.eos_index
            return ids

        def encode2d(info, shape, append_eos, consumer, padding=True, is_indice=False):
            """
            :param info:
            :param shape:
            :param append_eos:
            :param consumer:
            :return:
            """
            # dim0, dim1 = shape
            ids = []
            for i in range(0, shape):
                ids_ = encode(info[i], len(info[i]), append_eos, consumer).numpy().tolist()
                ids.append(ids_)
            if padding:
                return padding_data(ids, is_indice)  # padded data (type: tensor)
            return ids  # non-padding data (type:list)

        def reverse(nodes, indices):
            reversed_node = []
            for indice in indices:
                reversed = []
                for item in indice:
                    if item < len(nodes):
                        reversed.append(nodes[item].numpy().tolist())
                    else:
                        reversed.append(self.pad4amr())
                reversed_node.append(reversed)
            return reversed_node

        node_ids = encode(nodes[:nnodes], nnodes, append_eos, consumer)
        edges_ids = encode(edges[:nnodes], nnodes, append_eos, consumer)
        in_neigh_edge_ids = padding_data(in_neigh_edges[:nnodes], max_in_neighbor, is_indice=True)
        out_neigh_edge_ids = padding_data(out_neigh_edges[:nnodes], max_out_neighbor, is_indice=True)
        in_neigh_indice_ids = padding_data(in_neigh_indices[:nnodes], max_in_neighbor, is_indice=True)
        out_neigh_indice_ids = padding_data(out_neigh_indices[:nnodes], max_out_neighbor, is_indice=True)
        assert out_neigh_indice_ids.shape == out_neigh_indice_ids.shape, "Both out indices and out edges must have the same dims."
        assert in_neigh_indice_ids.shape == in_neigh_indice_ids.shape, "Both in indices and in edges must have the same dims."
        return node_ids, edges_ids, in_neigh_edge_ids, in_neigh_indice_ids, out_neigh_edge_ids, out_neigh_indice_ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
            filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        is_amr = filename.split(".")[-1] == "amr"
        if is_amr:
            node_counter = Counter()
            edge_counter = Counter()
        else:
            counter = Counter()
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            i = 0
            while line:
                if is_amr:
                    i += 1
                    # print(str(i) + ": " + line)
                    nodes, edges, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, max_node, max_in_neigh, max_out_neigh, max_sent = data_utils.read_amr(
                        line)
                    node_counter.update(nodes)
                    edge_counter.update(edges)
                    # for e in in_neigh_edges:
                    #     edge_counter.update(e)
                    # for e in out_neigh_edges:
                    #     edge_counter.update(e)
                else:
                    for word in tokenize(line):
                        counter.update([word])
                    counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        if is_amr:
            return node_counter, edge_counter
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        is_amr = filename.split(".")[-1] == "amr"

        def merge_result(counter):
            if is_amr:
                # node_dict, edge_dict = dict
                node_counter, edge_counter = counter
                for w, c in sorted(node_counter.items()):
                    dict.add_symbol(w, c)
                for w, c in sorted(edge_counter.items()):
                    dict.add_symbol(w, c)
            else:
                for w, c in sorted(counter.items()):
                    dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word
                )
            )


class TruncatedDictionary(object):
    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {},
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
