from fairseq.data.data_utils import read_amr
from collections import Counter
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "./tmp/train_.amr"
    counter = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            amr_node, edges, in_indices, in_edges, out_indices, out_edges, \
            max_node, max_in_neigh, max_out_neigh, max_sent = read_amr(line.lower())
            counter.append(amr_node)

    freq = list(c.values())
    freq = [x for x in freq if x <= 100]
    plt.plot(freq)
    plt.show()