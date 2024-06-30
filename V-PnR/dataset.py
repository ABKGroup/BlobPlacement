import re
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
import resource

from torch_geometric.data import Batch, Dataset, Data
from scipy.sparse import coo_matrix, lil_matrix

def custom_collate_fn(batch):
    data_list = [data for data in batch]
    batch_data = Batch.from_data_list(data_list)
    return batch_data

class ClusterDataset(Dataset):
    def __init__(self, data_dir, exp_dir, train_dir, test_dir, num_features, pickle_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.exp_dir = exp_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.nfeatures = num_features
        self.pickle_path = pickle_path 

        if not os.path.exists(self.pickle_path):
            self._process_and_save()

        print('Loading data...')
        with open(self.pickle_path, 'rb') as f:
            self.data = pickle.load(f)

    def _process_and_save(self, norm="minmax", norm_data="./model.norm"):
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        print('Building file list...')
        file_list = self._build_file_list(self.train_dir, self.test_dir)

        print('Processing data...')
        mp.set_start_method('spawn')
        with mp.Pool(processes=100) as pool:
            file_groups = [file_list[i:i+100] for i in range(0, len(file_list), 100)]

            for idx, file_group in enumerate(file_groups):
                print(f'Processing {len(file_group)} files in group {idx + 1}/{len(file_groups)}')
                result = pool.map(self._process_raw_data, file_group)
                with open(f'../data_dir/result_{idx}.pkl', 'wb') as f:
                    pickle.dump(result, f)

        processed_data = []
        for idx in range(len(file_groups)):
            with open(f'../data_dir/result_{idx}.pkl', 'rb') as f:
                result = pickle.load(f)
                processed_data.extend(result)

        print('Normalizing data...')
        if norm == "minmax":
            if os.path.exists(norm_data):
                print('Reading normalization data...')
                data_dict = {}
                with open(norm_data, 'r') as f:
                    for line in f.readlines():
                        sp = line.split(',')
                        k = sp[0]
                        v = [float(x) for x in sp[1:]]
                        data_dict[k] = v 
                    min_x = torch.tensor(data_dict["min_x"])
                    max_x = torch.tensor(data_dict["max_x"])
                    min_y = torch.tensor(data_dict["min_y"])
                    max_y = torch.tensor(data_dict["max_y"])

            else: 
                min_x, max_x = torch.tensor(float('inf')), torch.tensor(float('-inf'))
                min_y, max_y = torch.tensor(float('inf')), torch.tensor(float('-inf'))

                for data in processed_data:
                    min_x = torch.min(min_x, data.x.min(dim=0).values)
                    max_x = torch.max(max_x, data.x.max(dim=0).values)
                    min_y = torch.min(min_y, data.y.min(dim=0).values)
                    max_y = torch.max(max_y, data.y.max(dim=0).values)
                    
            for data in processed_data:
                data.x = (data.x - min_x) / (max_x - min_x)
                data.y = (data.y - min_y) / (max_y - min_y)
        elif norm == "std":
            sum_x, sq_sum_x, sum_y, sq_sum_y = torch.zeros(self.nfeatures, dtype=data.x.dtype), torch.zeros(self.nfeatures, dtype=data.x.dtype), torch.zeros(1), torch.zeros(1)
            count = 0

            for data in processed_data:
                sum_x += data.x.sum(dim=0)
                sq_sum_x += (data.x ** 2).sum(dim=0)
                sum_y += data.y.sum(dim=0)
                sq_sum_y += (data.y ** 2).sum(dim=0)
                count += data.x.size(0)

            mean_x = sum_x / count
            std_x = (sq_sum_x / count - mean_x ** 2).sqrt()
            mean_y = sum_y / count
            std_y = (sq_sum_y / count - mean_y ** 2).sqrt()

            for data in processed_data:
                data.x = (data.x - mean_x) / std_x
                data.y = (data.y - mean_y) / std_y

        print('Dumping data...')
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(processed_data, f)

    def _process_raw_data(self, args):
        node_path, edge_path, label_path, flag = args
        node_data = pd.read_csv(node_path).drop('height', axis=1)
        node_features = node_data.iloc[:, 1:].values

        edge_data = pd.read_csv(edge_path)
        edges, connected_nodes = edge_data['hyperedge'].tolist(), [nodes.split('-') for nodes in edge_data['connected_nodes']]
        unique_nodes = sorted(list(set().union(*connected_nodes)))
        num_nodes = len(unique_nodes)
        incidence_matrix = lil_matrix((num_nodes, len(edges)), dtype=int)
        for col, nodes in enumerate(connected_nodes):
            for node in nodes:
                incidence_matrix[unique_nodes.index(node), col] = 1
        coo_incidence_matrix = incidence_matrix.tocoo()
        hyperedge_index = torch.tensor(np.vstack((coo_incidence_matrix.row, coo_incidence_matrix.col)), dtype=torch.long)

        label_data = pd.read_csv(label_path)

        hpwl = label_data['hwpl'].iloc[0]
        top10 = label_data['top10'].iloc[0]
        width = label_data['width'].iloc[0] / 2000
        height = label_data['height'].iloc[0] / 2000

        num_nets = len(edge_data)
        normalized_hpwl = hpwl / num_nets
        avg_normalized_hpwl = normalized_hpwl / (width + height)
        label = avg_normalized_hpwl * 0.1 + top10

        x, y = torch.tensor(node_features, dtype=torch.float), torch.tensor(label, dtype=torch.float)
        data = Data(x=x, y=y, hyperedge_index=hyperedge_index)

        design = node_path.split('/')[-4].split('_1.0')[0]
        cluster_idx = node_path.split('/')[-1].split('.')[0].split('_')[1]
        util = node_path.split('/')[-2].split('_')[0]
        ar = node_path.split('/')[-2].split('_')[1]

        data.design = design
        data.cluster_idx = cluster_idx
        data.util = util
        data.ar = ar 
        data.flag = flag
       
        data.hpwl = avg_normalized_hpwl
        data.cong = top10

        return data

    @staticmethod
    def _build_file_list(train_dir, test_dir):
        file_list = []
        for run in os.listdir(train_dir):
            if not os.path.exists(os.path.join(train_dir, run, 'feature')) : continue
            for comb in os.listdir(os.path.join(train_dir, run, 'feature')):
                fea_dir = os.path.join(train_dir, run, 'feature', comb)
                lab_dir = os.path.join(train_dir, run, 'label', comb)


                node_pattern = r'cluster_(\d+).node'

                idx = []
                for filename in os.listdir(fea_dir):
                    match = re.match(node_pattern, filename)
                    if match:
                        idx.append(int(match.group(1)))

                if idx :
                    for i in idx:
                        node_path = os.path.join(train_dir, run, 'feature', comb, f'cluster_{i}.node')
                        edge_path = os.path.join(train_dir, run, 'feature', comb, f'cluster_{i}.edge')
                        label_path = os.path.join(train_dir, run, 'label', comb, f'cluster_{i}.label')

                        if os.path.exists(node_path) and os.path.exists(edge_path) and os.path.exists(label_path):
                            file_list.append((node_path, edge_path, label_path, "train"))
                        else :
                            sys.exit(f'ERROR:No file (node_path {node_path} / edge_path {edge_path} / label_path {label_path}')

        for run in os.listdir(test_dir):
            for comb in os.listdir(os.path.join(test_dir, run, 'feature')):
                fea_dir = os.path.join(test_dir, run, 'feature', comb)
                lab_dir = os.path.join(test_dir, run, 'label', comb)

                node_pattern = r'cluster_(\d+).node'

                idx = []
                for filename in os.listdir(fea_dir):
                    match = re.match(node_pattern, filename)
                    if match:
                        idx.append(int(match.group(1)))

                if idx :
                    for i in idx:
                        node_path = os.path.join(test_dir, run, 'feature', comb, f'cluster_{i}.node')
                        edge_path = os.path.join(test_dir, run, 'feature', comb, f'cluster_{i}.edge')
                        label_path = os.path.join(test_dir, run, 'label', comb, f'cluster_{i}.label')

                        if os.path.exists(node_path) and os.path.exists(edge_path) and os.path.exists(label_path):
                            file_list.append((node_path, edge_path, label_path, "test"))
                        else :
                            sys.exit(f'ERROR:No file (node_path {node_path} / edge_path {edge_path} / label_path {label_path}')
        
        return file_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
