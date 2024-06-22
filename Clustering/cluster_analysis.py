import csv
import numpy as np
import networkx as nx
from scipy.linalg import eigh
import os
from tqdm import tqdm
from scipy.sparse.linalg import eigs

node_id_to_index = {}
node_index_to_id = {}

class UF:
    def __init__(self, f_v) -> None:
        self.parent = {}
        self.rank = f_v
        for i in range(len(f_v)):
            self.parent[i] = i  # every node is parent of itself
        self.count = len(f_v)
    
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i =  self.parent[i] # keep going up the tree
        return i
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[y_root] = x_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            if x_root < y_root:
                self.parent[y_root] = x_root
            else:
                self.parent[x_root] = y_root
        self.count -= 1
    
def get_hks(L, K, ts):
    """
    From https://github.com/ctralie/pyhks/blob/master/hks.py
    ----------
    L : Graph Laplacian

    K : int
        Number of eigenvalues/eigenvectors to use
    ts : ndarray (T)
        The time scales at which to compute the HKS
    
    Returns
    -------
    hks : ndarray (N, T)
        A array of the heat kernel signatures at each of N points
        at T time intervals
    """
    print("Computing HKS")
    (eigvalues, eigvectors) = eigs(L, k = 20)
    print("Done computing HKS")
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None] * ts.flatten()[None, None, :])
    return np.sum(res, 1)


def get_hks_filtration(g, edges):
    num_nodes = len(g.nodes())
    graph_laplacian = nx.normalized_laplacian_matrix(g).toarray().astype(float)
    hks = get_hks(graph_laplacian, num_nodes, ts=np.array([1, 10]))
    f_v = -hks[:, -1]
    f_e = [max(f_v[u], f_v[v]) for u, v in edges]
    return f_v, f_e

def parse_edge_csv_file(fname):
    """
    Parses the edge csv file and returns a list of edges
    """
    g = nx.Graph()
    edges = set()
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['Source']
            sink = row['Sink']
            wt = float(row['Weight'])
            if src not in node_id_to_index:
                node_id_to_index[src] = len(node_id_to_index)
                node_index_to_id[node_id_to_index[src]] = src
            if sink not in node_id_to_index:
                node_id_to_index[sink] = len(node_id_to_index)
                node_index_to_id[node_id_to_index[sink]] = sink
            g.add_edge(node_id_to_index[src], node_id_to_index[sink], weight=wt)
            u, v = tuple(sorted([node_id_to_index[src], node_id_to_index[sink]]))
            edges.add((u, v))
    return g, list(edges)

def get_clusters(out_filename, edge_filename, persistence_threshold=0.1):
    g, edges = parse_edge_csv_file(edge_filename)
    f_v, f_e = get_hks_filtration(g, edges)
    edge_idx = np.argsort(f_e)
    uf_pers = UF(f_v)
    uf_reg = UF(f_v)
    
    for e in tqdm(edge_idx):
        u, v = edges[e]
        u_root = uf_pers.find(u)
        v_root = uf_pers.find(v)
        if u_root == v_root:
            continue
        pers = f_e[e] - max([f_v[u_root], f_v[v_root]])
        if pers < persistence_threshold:
            uf_reg.union(u, v)
        uf_pers.union(u_root, v_root)
    cluster_id = {}
    print(f'Number of clusters: {uf_reg.count}')
    
    
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Cluster_id'])
        for i in range(len(f_v)):
            cluster_id_raw = uf_reg.find(i)
            if cluster_id_raw not in cluster_id:
                cluster_id[cluster_id_raw] = len(cluster_id)
            writer.writerow([node_index_to_id[i], cluster_id[cluster_id_raw]])
    print(f'Wrote clusters to {out_filename}')

if __name__ == '__main__':
    pers_thresh = 0.001
    edge_file_name = '/home/fetzfs_projects/PlacementCluster/sakundu/PhysicalSynthesis/run_megaboom/MegaBoom_edges.csv'
    edge_file_name = '/home/fetzfs_projects/PlacementCluster/sakundu/PhysicalSynthesis/run_aes/aes_cipher_top_edges.csv'
    get_clusters(f'aes_cipher_top_{pers_thresh}.csv', edge_file_name, persistence_threshold=pers_thresh)
    # edge_file_name = 'test_edges.csv'
    # for pers_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #     get_clusters(f'MegaBoom_cluster_{pers_thresh}.csv', edge_file_name, persistence_threshold=pers_thresh)


