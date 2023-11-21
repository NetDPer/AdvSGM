import random
import networkx as nx
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--indep_run_times', default=2)
parser.add_argument('--n_epoch', default=2000)
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='uniform', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--lr', default=0.1)
parser.add_argument('--K', default=5)
parser.add_argument('--sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=1)
parser.add_argument('--is_GradientClip', default=True)
parser.add_argument('--clip_value', default=1)

args = parser.parse_args()

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

class prepare_data:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def prepare_data_for_dis(self, edge):
        global edge_batch_index, negative_node
        #
        # if args.edge_sampling == 'numpy':
        #     edge_batch_index = np.random.choice(self.num_of_edges, size=args.batch_size, p=self.edge_distribution)
        # elif args.edge_sampling == 'atlas':
        #     edge_batch_index = self.edge_sampling.sampling(args.batch_size)
        # elif args.edge_sampling == 'uniform':
        #     edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.batch_size)

        u_i = []
        u_j = []
        label = []
        # for edge_index in edge_batch_index:
        #     edge = self.edges[edge_index]
        # for edge in self.edges:
        if self.g.__class__ == nx.Graph:
            if np.random.rand() > 0.5:
                edge = (edge[1], edge[0])
        u_i.append(edge[0])
        u_j.append(edge[1])
        label.append(1)
        for i in range(args.K):
            while True:
                if args.node_sampling == 'numpy':
                    negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                elif args.node_sampling == 'atlas':
                    negative_node = self.node_sampling.sampling()
                elif args.node_sampling == 'uniform':
                    negative_node = np.random.randint(0, self.num_of_nodes)
                if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                    break
            u_i.append(edge[0])
            u_j.append(negative_node)
            label.append(-1)

        return u_i, u_j, label

def loadGraphFromEdgeListTxt(file_name, directed=True):
    with open(file_name, 'r') as f:
        # n_nodes = f.readline()
        # f.readline() # Discard the number of edges
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
    return G

# ui_uj_label_dict = {}
# key_index = 0
# for edge in data_loader.edges:
#     ui_uj_label_list = []
#     # ui_uj_label_tuple = ()
#     u_i, u_j, label = data_loader.prepare_data_for_dis(edge)
#     for index in range(len(u_i)):
#         ui_uj_label_list.append((u_i[index], u_j[index], label[index]))
#
#     ui_uj_label_dict[key_index] = ui_uj_label_list
#     key_index = key_index + 1
#     # print(u_i)
#     # print(u_j)
#     # print(label)

# # take all keys in dict to form a list
# ui_uj_label_dict.keys()
# print(ui_uj_label_dict.keys())
# # randomly select n keys
# sampled_keys = random.sample(ui_uj_label_dict.keys(), args.batch_size)
#
# u_i = []
# u_j = []
# label = []
# for key in sampled_keys:
#     ui_uj_label_list = ui_uj_label_dict[key]
#     for index in ui_uj_label_list:
#         print(index)
#         u_i.append(index[0])
#         u_j.append(index[1])
#         label.append(index[2])
#
# print(123)
# # take the values according to keys

def graph_to_subgraph_set(graph):
    data_loader = prepare_data(graph)
    ui_uj_label_dict = {}
    key_index = 0
    for edge in data_loader.edges:
        # print(len(data_loader.edges), key_index)
        ui_uj_label_list = []
        # ui_uj_label_tuple = ()
        u_i, u_j, label = data_loader.prepare_data_for_dis(edge)
        for index in range(len(u_i)):
            ui_uj_label_list.append((u_i[index], u_j[index], label[index]))

        ui_uj_label_dict[key_index] = ui_uj_label_list
        key_index = key_index + 1
        # print(u_i)
        # print(u_j)
        # print(label)

    return ui_uj_label_dict

def batchSample_from_subgra_set(subgra_set, batch_size):
    ui_uj_label_dict = subgra_set
    # take all keys in dict to form a list
    ui_uj_label_dict.keys()
    # print(ui_uj_label_dict.keys())
    # randomly select n keys
    sampled_keys = random.sample(ui_uj_label_dict.keys(), batch_size)

    u_i = []
    u_j = []
    label = []
    for key in sampled_keys:
        ui_uj_label_list = ui_uj_label_dict[key]
        for index in ui_uj_label_list:
            # print(index)
            u_i.append(index[0])
            u_j.append(index[1])
            label.append(index[2])

    return u_i, u_j, label

if __name__ == '__main__':
    set_dataset_name = 'lp_PPI'
    set_split_name = 'train0.8_test0.2'

    oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
    train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

    # Load graph
    Graph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

    # data_loader = prepare_data(Graph)

    subgra_set = graph_to_subgraph_set(Graph)

    u_i, u_j, label = batchSample_from_subgra_set(subgra_set, args.batch_size)

'''
triangle_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)] 
triangle_dict = {'triangles': triangle_list}
'''

'''
import numpy as np

def generate_random_batch_with_poisson_subsampling(values, L):
    N = len(values)  # Total number of elements
    sampling_prob = L / N  # Calculate the sampling probability

    # Generate Poisson-distributed samples for subsampling
    poisson_samples = np.random.poisson(sampling_prob, N)

    # Create the random batch Lt with Poisson subsampling
    Lt_indices = [i for i in range(N) if poisson_samples[i] == 1][:L]

    # Return the randomly sampled batch Lt
    return [values[i] for i in Lt_indices]

# Example usage
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Your list of values
L = 3  # Desired batch size

random_batch = generate_random_batch_with_poisson_subsampling(values, L)
print("Random Batch Lt:", random_batch)
'''
