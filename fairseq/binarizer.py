# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch
from fairseq.data.data_utils import read_amr


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod
    def binarize(
            filename,
            dict,
            consumer,
            tokenize=tokenize_line,
            append_eos=True,
            reverse_order=False,
            offset=0,
            end=-1,
            already_numberized=False,
            is_amr=False
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if not is_amr:
                    if end > 0 and f.tell() > end:
                        break
                    if already_numberized:
                        id_strings = line.strip().split()
                        id_list = [int(id_string) for id_string in id_strings]
                        if reverse_order:
                            id_list.reverse()
                        if append_eos:
                            id_list.append(dict.eos())
                        ids = torch.IntTensor(id_list)
                    else:
                        ids = dict.encode_line(
                            line=line,
                            line_tokenizer=tokenize,
                            add_if_not_exist=False,
                            consumer=replaced_consumer,
                            append_eos=append_eos,
                            reverse_order=reverse_order,
                        )
                        ids = [ids]
                else:
                    ids = dict.encode_graph_info(line, add_if_not_exist=False, consumer=replaced_consumer)
                    # node_ids, in_neigh_edge_ids, in_neigh_indice_ids, out_neigh_edge_ids, out_neigh_indice_ids = ids
                    # ids = (node_ids

                nseq += 1
                ntok += len(ids[0])
                for ids_ in ids:
                    consumer(ids_)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
