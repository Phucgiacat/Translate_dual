from fairseq.data.data_utils import read_amr
from collections import Counter
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "./ori_data/amr/tst2013.en.pred.anonymized"
    counter = []
    counter_edge = Counter()
    counter_node = Counter()
    max_node = 0
    min_node = 999
    max_parent = 0
    max_child = 0
    min_parent = 999
    min_child = 999
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if sum(x == '(' for x in line) == sum(x == ')' for x in line):
                amr_node, edges, in_indices, in_edges, out_indices, out_edges, \
                max_node, max_in_neigh, max_out_neigh, max_sent = read_amr(line.lower())
                counter_node.update(amr_node)
                counter_edge.update(edges)
                max_node = len(amr_node) if len(amr_node) > max_node else max_node
                min_node = len(amr_node) if len(amr_node) < min_node else min_node

                for indice in in_indices:
                    max_parent = len(indice) if len(indice) > max_parent else max_parent
                    min_parent = len(indice) if len(indice) < min_parent else min_parent
                for indice in out_indices:
                    max_child = len(indice) if len(indice) > max_child else max_child
                    min_child = len(indice) if len(indice) < min_child else min_child


    for x in sorted(counter_node, key=counter_node.get)[:10]:
        print(x + "& " + str(counter_node[x]))
    for x in sorted(counter_edge, key=counter_edge.get)[:10]:
        print(x + "& " + str(counter_edge[x]))
    freq = list(c.values())
    freq = [x for x in freq if x <= 100]
    plt.plot(freq)
    plt.show()
