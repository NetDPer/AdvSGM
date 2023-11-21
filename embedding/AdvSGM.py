import time
import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
import graph_util
import math
from sklearn import metrics
from sklearn.externals import joblib
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
import save_to_excel
import operator
from sklearn.cluster import AffinityPropagation
import preprocess_graph

parser = argparse.ArgumentParser()
parser.add_argument('--indep_run_times', default=5)
parser.add_argument('--n_epoch', default=5)
parser.add_argument('--embedding_dim', default=128, help='16 for node clustering, 128 for link prediction')
parser.add_argument('--d_batch_size', default=512, help='128 for node clustering, 512 for link prediction')
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='uniform', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--d_epoch', default=25)
parser.add_argument('--g_epoch', default=5)
parser.add_argument('--g_batch_size', default=512, help='128 for node clustering, 512 for link prediction')
parser.add_argument('--lr_gen', default=0.1)
parser.add_argument('--lr_dis', default=0.1)
parser.add_argument('--K', default=5)
parser.add_argument('--low_bound', default=10**(-5), help='low bound for exp func')
parser.add_argument('--upper_bound', default=120, help='upper bound for exp func')
parser.add_argument('--dis_sigma', default=2)
parser.add_argument('--gen_sigma', default=2)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--dis_sens', default=1, help='sensitivity for discriminator')
parser.add_argument('--epsilon', default=1)
parser.add_argument('--s', default=6, help='normalization weight parameter')

args = parser.parse_args()

class discriminator:
    def __init__(self, args, edge_distribution):
        with tf.variable_scope('forward_pass'):
            self.edge_distribution = edge_distribution
            self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[None])
            self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[None])
            self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[None])
            self.optimal_val_ij = tf.placeholder(name='optimal_values', dtype=tf.float32,
                                                 shape=[None])
            self.fake_u_i_embedding = tf.placeholder(name='fake_u_i_emb', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.fake_u_j_embedding = tf.placeholder(name='fake_u_j_emb', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.Gaussian_noise_i = tf.placeholder(name='Gau_noise_i', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.Gaussian_noise_j = tf.placeholder(name='Gau_noise_j', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])

            self.embedding = tf.get_variable('target_emb', [args.num_of_nodes, args.embedding_dim],
                                             initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)

            self.embedding = self.embedding / [tf.norm(self.embedding) * args.s]

            if args.proximity == 'first-order':
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
            elif args.proximity == 'second-order':
                self.context_embedding = tf.get_variable('context_emb', [args.num_of_nodes, args.embedding_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

                self.context_embedding = self.context_embedding / [tf.norm(self.context_embedding) * args.s]

            self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)

            sigmoid_test = tf.div(1.0, 1 + expclip(-self.label * self.inner_product,
                                                    args.low_bound, args.upper_bound))

            self.sgm_loss = -tf.log(sigmoid_test)
            # self.sgm_loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))

            # ----------- adv term ------------
            self.adv_inner_product_1 = tf.reduce_sum(self.u_i_embedding * self.fake_u_i_embedding
                                                     + self.u_i_embedding * self.Gaussian_noise_i, axis=1)
            sigmoid_i = tf.div(1.0, 1 + expclip(-self.adv_inner_product_1,
                                                    args.low_bound, args.upper_bound))
            # sigmoid_i = tf.sigmoid(self.adv_inner_product_1)

            self.adv_inner_product_2 = tf.reduce_sum(self.u_j_embedding * self.fake_u_j_embedding
                                                     + self.u_j_embedding * self.Gaussian_noise_j, axis=1)
            sigmoid_j = tf.div(1.0, 1 + expclip(-self.adv_inner_product_2,
                                                    args.low_bound, args.upper_bound))
            # sigmoid_j = tf.sigmoid(self.adv_inner_product_2)

            self.weight_i = 1 / sigmoid_i
            self.weight_j = 1 / sigmoid_j

            self.adv_loss = self.weight_i * tf.log(1 - sigmoid_i) \
                            + self.weight_j * tf.log(1 - sigmoid_j)

            self.loss = tf.reduce_mean(self.sgm_loss + self.adv_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr_dis)
            self.train_op = self.optimizer.minimize(self.loss)

class generator:
    def __init__(self, args):
        self.gen_W_1 = tf.get_variable(name='gen_W_i', dtype=tf.float32,
                                       shape=[args.num_of_nodes, args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_B_1 = tf.get_variable(name='gen_B_i', dtype=tf.float32,
                                       shape=[args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_W_2 = tf.get_variable(name='gen_W_j', dtype=tf.float32,
                                       shape=[args.num_of_nodes, args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_B_2 = tf.get_variable(name='gen_B_j', dtype=tf.float32,
                                       shape=[args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)

        self.node_ids = tf.placeholder(tf.int32, shape=[None])

        self.noise_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # noise matrix

        self.pos_node_i_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # pos samples from dis

        self.pos_node_j_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # pos samples from dis

        self.gen_w_1 = tf.matmul(tf.one_hot(self.node_ids, depth=args.num_of_nodes), self.gen_W_1)

        self.gen_w_2 = tf.matmul(tf.one_hot(self.node_ids, depth=args.num_of_nodes), self.gen_W_2)

        test_1 = self.noise_embedding * self.gen_w_1 + self.gen_B_1

        # self.fake_embedding_1 = tf.nn.leaky_relu(test_1)

        self.fake_embedding_1 = tf.nn.sigmoid(test_1)

        test_2 = self.noise_embedding * self.gen_w_2 + self.gen_B_2

        # self.fake_embedding_2 = tf.nn.leaky_relu(test_2)

        self.fake_embedding_2 = tf.nn.sigmoid(test_2)

        self.gen_term_1 = self.fake_embedding_1 * self.pos_node_i_embedding \
                          + self.pos_node_i_embedding * self.noise_embedding
        self.gen_term_2 = self.fake_embedding_2 * self.pos_node_j_embedding \
                          + self.pos_node_j_embedding * self.noise_embedding
        self.loss = tf.reduce_mean(tf.log(1 - tf.log_sigmoid(self.gen_term_1)) + tf.log(1 - tf.log_sigmoid(self.gen_term_2)))

        self.optimizer = tf.train.AdamOptimizer(args.lr_gen)

        self.g_updates = self.optimizer.minimize(self.loss)

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

    def prepare_data_for_dis(self, sess, generator):
        global edge_batch_index, negative_node

        if args.edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=args.d_batch_size, p=self.edge_distribution)
        elif args.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(args.d_batch_size)
        elif args.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.d_batch_size)

        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
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

        Gaussian_noise_i = np.random.normal(loc=0, scale=args.dis_sens * args.dis_sigma, size=(len(u_i), args.embedding_dim))
        Gaussian_noise_j = np.random.normal(loc=0, scale=args.dis_sens * args.dis_sigma, size=(len(u_i), args.embedding_dim))

        fake_u_i_embedding = sess.run(generator.gen_w_1, feed_dict={generator.node_ids: u_i})
        fake_u_j_embedding = sess.run(generator.gen_w_2, feed_dict={generator.node_ids: u_j})

        return u_i, u_j, label, fake_u_i_embedding, fake_u_j_embedding, Gaussian_noise_i, Gaussian_noise_j

    def prepare_data_for_gen(self, sess, model, index, node_list, batch_size_g):
        node_ids = []
        if isinstance(node_list, list) is False:
            node_list = list(node_list)

        for node_id in node_list[index * batch_size_g: (index + 1) * batch_size_g]:
            node_ids.append(node_id)

        GaussianNoise_embedding = np.random.normal(loc=0, scale=args.gen_sigma, size=(len(node_ids), args.embedding_dim))

        node_i_embedding = sess.run(model.u_i_embedding, feed_dict={model.u_i: node_ids})
        node_j_embedding = sess.run(model.u_j_embedding, feed_dict={model.u_j: node_ids})

        return node_ids, GaussianNoise_embedding, node_i_embedding, node_j_embedding

def expclip(x, a=None, b=None):
    '''
    clipping exp function to limit Sigmoid.
    Exponential soft clipping, with parameterized corner sharpness.
    '''
    # default scaling constants to match tanh corner shape
    _c_tanh = 2 / (np.e * np.e + 1)  # == 1 - np.tanh(1) ~= 0.24
    _c_softclip = np.log(2) / _c_tanh
    _c_expclip = 1 / (2 * _c_tanh)

    c = _c_expclip
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = tf.clip_by_value(x, a, b)

    if a is not None:
        v = v + tf.exp(-c * np.abs(x - a)) / (2 * c)
    if b is not None:
        v = v - tf.exp(-c * np.abs(x - b)) / (2 * c)

    return v

class trainModel:
    def __init__(self, inf_display, graph, test_pos=None, test_neg=None, node_label=None):
        self.inf_display = inf_display
        self.node_label = node_label
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.graph = graph
        self.data_loader = prepare_data(self.graph)
        args.num_of_nodes = self.data_loader.num_of_nodes
        args.num_of_edges = self.data_loader.num_of_edges
        self.model = discriminator(args, self.data_loader.edge_distribution)
        self.generator = generator(args)

    def train_dis(self, test_task=None, test_ratios=None, output_filename=None):
        # suffix = args.proximity
        best_auc = []
        best_MI = []
        best_AMI = []
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        for indep_run_time in range(args.indep_run_times):
            subgra_set = preprocess_graph.graph_to_subgraph_set(self.graph)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # note that this initilization's location
                flag_auc = 0
                flag_MI = 0
                flag_AMI = 0
                # orders for RDP
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = np.zeros_like(orders, dtype=float)
                for epoch in range(args.n_epoch):
                    for d_epoch in range(args.d_epoch):
                        u_i, u_j, label = preprocess_graph.batchSample_from_subgra_set(subgra_set, args.d_batch_size)
                        Gaussian_noise_i = np.random.normal(loc=0, scale=args.dis_sens * args.dis_sigma,
                                                            size=(len(u_i), args.embedding_dim))
                        Gaussian_noise_j = np.random.normal(loc=0, scale=args.dis_sens * args.dis_sigma,
                                                            size=(len(u_i), args.embedding_dim))
                        fake_u_i_embedding = sess.run(self.generator.gen_w_1, feed_dict={self.generator.node_ids: u_i})
                        fake_u_j_embedding = sess.run(self.generator.gen_w_2, feed_dict={self.generator.node_ids: u_j})

                        # # for index in range(math.floor(args.num_of_edges / args.d_batch_size)):
                        # u_i, u_j, label, fake_u_i_embedding, fake_u_j_embedding, \
                        # Gaussian_noise_i, Gaussian_noise_j = self.data_loader.prepare_data_for_dis(sess, self.generator)

                        feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                                     self.model.fake_u_i_embedding: fake_u_i_embedding,
                                     self.model.fake_u_j_embedding: fake_u_j_embedding,
                                     self.model.Gaussian_noise_i: Gaussian_noise_i,
                                     self.model.Gaussian_noise_j: Gaussian_noise_j}
                        _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                        # --------- RDP mechanism -------------
                        sampling_prob = args.d_batch_size / args.num_of_edges

                        iter_steps = args.n_epoch * args.d_epoch

                        rdp += compute_rdp(sampling_prob, args.dis_sigma, iter_steps, orders)

                        _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=args.epsilon)
                        print(_eps, _delta)

                        if _delta > args.delta:
                            print('jump out')
                            break

                        embedding = sess.run(self.model.embedding)
                        # --------------------------------------------------------------------------------
                        if test_task == 'lp':
                            embedding_mat = np.dot(embedding, embedding.T)
                            auc = CalcAUC(embedding_mat, self.test_pos, self.test_neg)
                            if auc > flag_auc:
                                flag_auc = auc
                            print(indep_run_time, epoch, d_epoch, loss, auc)

                    if test_task == 'ncluster':
                        MI, AMI = evaluate(node_label, embedding)
                        if MI > flag_MI:
                            flag_MI = MI
                        if AMI > flag_AMI:
                            flag_AMI = AMI
                        print(indep_run_time, epoch, d_epoch, loss, MI)

                    gen_loss = 0.0
                    gen_neg_loss = 0.0
                    gen_cnt = 0
                    for g_epoch in range(args.g_epoch):
                        gen_loss, gen_neg_loss, gen_cnt = self.train_gen(sess, gen_loss, gen_neg_loss, gen_cnt)

                if test_task == 'lp':
                    best_auc.append(flag_auc)

                if test_task == 'ncluster':
                    best_MI.append(flag_MI)
                    best_AMI.append(flag_AMI)

        # save and write
        mark_time = str(time.time()).split(".")[0]
        output_final_name = output_filename + '_' + mark_time + '.xlsx'

        if test_task == 'lp':
            # pd.DataFrame(best_auc).to_excel(output_final_name, index=False, header=['auc'])
            save_to_excel.output_to_excel(best_auc, 'AUC', output_final_name)

        if test_task == 'ncluster':
            save_to_excel.output_to_excel(best_MI, 'MI', output_final_name)

    def train_gen(self, sess, gen_loss, neg_loss, gen_cnt):
        node_list = self.graph.nodes()
        batch_size_g = args.g_batch_size * (args.K + 1)
        for index in range(math.floor(len(node_list) / batch_size_g)):
            node_ids, noise_embedding, node_i_embedding, node_j_embedding \
                = self.data_loader.prepare_data_for_gen(sess, self.model, index, node_list, batch_size_g)
            _loss, _neg_loss = sess.run(
                [self.generator.g_updates, self.generator.loss],
                feed_dict={self.generator.node_ids: np.array(node_ids),
                           self.generator.noise_embedding: np.array(noise_embedding),
                           self.generator.pos_node_i_embedding: np.array(node_i_embedding),
                           self.generator.pos_node_j_embedding: np.array(node_j_embedding)})

        return gen_loss, neg_loss, gen_cnt

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def compute_delta(steps, batch_size, dataset_size, target_eps):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / dataset_size
    # compute RDP of the Sampled Gaussian mechanism for each order
    rdp = compute_rdp(sampling_probability, args.dis_sigma, steps, orders)
    # compute epsilon given the list of RDP values and the target delta
    return get_privacy_spent(orders, rdp, target_eps=target_eps)

def get_HT(X, Y, label_pred, name='k_means'):
    score_funcs = [
        metrics.adjusted_mutual_info_score,
        metrics.mutual_info_score,
    ]

    km_scores_1 = metrics.mutual_info_score(Y, label_pred)
    km_scores_2 = metrics.adjusted_mutual_info_score(Y, label_pred)

    return km_scores_1, km_scores_2

def sortedDictValues1(adict):
    keys = list(adict.keys())
    keys.sort()
    values = list(map(adict.get, keys))
    return values

def evaluate(node_label, embedding_matrix):
    embedding_list = embedding_matrix.tolist()
    X = embedding_list
    Y = sortedDictValues1(node_label)
    ap = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15)
    ap.fit(X)
    label_pred = ap.labels_
    # print(label_pred)

    km_scores_1, km_scores_2 = get_HT(X, Y=Y, label_pred=label_pred, name='AffinityPropagation')
    return km_scores_1, km_scores_2

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

if __name__ == '__main__':
    test_task = 'lp'
    set_algo_name = 'AdvSGM_Vary_epsilon'

    args.s = args.K + 1
    args.dis_sens = (args.K + 1) / args.s
    '''
    'lp_PPI', n_epoch = 5
    'lp_wiki', n_epoch = 5
    'lp_BlogCatalog', n_epoch = 5
    'lp_facebook', n_epoch = 5
    'lp_arxiv' 15
    'lp_epinions' 25
    '''
    if test_task == 'lp':
        set_dataset_names = ['lp_PPI']
        set_split_name = 'train0.9_test0.1'

        epsilon_values = [1]
        for set_dataset_name in set_dataset_names:
            if set_dataset_name == 'lp_arxiv':
                args.n_epoch = 15
            if set_dataset_name == 'lp_epinions':
                args.n_epoch = 25
                args.indep_run_times = 2
            for epsilon_value in epsilon_values:
                tf.reset_default_graph()
                args.epsilon = epsilon_value
                set_epsilon_str = 'epsilon' + str(args.epsilon)
                set_learning_rate = 'step' + str(args.lr_dis)
                set_nepoch_name = 'nepoch' + str(args.n_epoch)
                # set_C = 'C' + str(args.clip_value)
                set_b = 'b' + str(args.upper_bound)
                set_K = 'K' + str(args.K)
                set_s = 's' + str(args.s)
                # ------------------------------------------------------
                oriGraph_filename = '../data/' + set_dataset_name +'/train_1'
                train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'
                output_filename = set_algo_name + '_' + set_dataset_name + '_' + set_split_name + '_' \
                                  + set_nepoch_name + '_' + set_epsilon_str + '_' + set_learning_rate + '_' \
                                  + '_' + set_b + '_' + set_K + '_' + set_s

                # Load graph
                trainGraph = graph_util.loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)
                trainGraph = nx.adjacency_matrix(trainGraph)
                # ------------------------------------------------------
                train_pos = joblib.load(train_filename + 'train_pos.pkl')
                train_neg = joblib.load(train_filename + 'train_neg.pkl')
                test_pos = joblib.load(train_filename + 'test_pos.pkl')
                test_neg = joblib.load(train_filename + 'test_neg.pkl')

                trainGraph = trainGraph.copy()  # the observed network
                trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
                trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
                trainGraph.eliminate_zeros()

                row, col = train_neg
                trainGraph = trainGraph.copy()
                trainGraph[row, col] = 1  # inject negative train
                trainGraph[col, row] = 1  # inject negative train
                trainGraph = nx.from_scipy_sparse_matrix(trainGraph)
                # ------------------------------------------------------
                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, test_pos=test_pos, test_neg=test_neg)
                tm.train_dis(test_task=test_task, output_filename=output_filename)

    if test_task == 'ncluster':
        set_dataset_names = ['ncluster_BlogCatalog']
        # set_dataset_names = ['ncluster_PPI', 'ncluster_wiki', 'ncluster_BlogCatalog']

        set_learning_rate = 'step' + str(args.lr_dis)
        epsilon_values = [3, 4, 5, 6]
        for set_dataset_name in set_dataset_names:
            for epsilon_value in epsilon_values:
                tf.reset_default_graph()
                args.epsilon = epsilon_value
                set_epsilon_str = 'epsilon' + str(args.epsilon)
                set_learning_rate = 'step' + str(args.lr_dis)
                set_nepoch_name = 'nepoch' + str(args.n_epoch)
                # set_C = 'C' + str(args.clip_value)
                set_b = 'b' + str(args.upper_bound)
                set_K = 'K' + str(args.K)
                set_s = 's' + str(args.s)

                train_filename = '../data/' + set_dataset_name + '/train_1'
                test_filename = '../data/' + set_dataset_name + '/test_1'
                output_filename = set_algo_name + '_' + set_dataset_name + '_' \
                                  + set_nepoch_name + '_' + set_epsilon_str + '_' + set_learning_rate + '_' \
                                  + '_' + set_b + '_' + set_K + '_' + set_s

                # Load graph
                G = loadGraphFromEdgeListTxt(train_filename, directed=False)

                node_label = {}
                with open(test_filename) as infile:
                    for line in infile.readlines():
                        if operator.contains(line, ','):
                            line = line.strip().split(',')
                        else:
                            line = line.strip().split()
                        s = int(line[0])
                        label = int(line[1])
                        node_label[s] = label

                print('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
                inf_display = [test_task, set_dataset_name]

                tm = trainModel(inf_display, G, node_label=node_label)
                tm.train_dis(test_task=test_task, test_ratios=[0], output_filename=output_filename)