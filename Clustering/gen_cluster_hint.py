import os
import sys
import re
import time
import networkx as nx
import igraph as ig
import leidenalg as la
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import community as community_louvain
from sklearn.metrics import pairwise_distances, davies_bouldin_score, \
    silhouette_score, calinski_harabasz_score


SEED = 42

def gen_rgb_color(num_colors:int) -> List[Tuple[int, int, int]]:
    # Generate an array of RGB values excluding 0 (black)
    rgb_values = np.linspace(50, 255, num=int(np.ceil(num_colors ** (1/3))),
                             dtype=int)

    # Generate a list of RGB colors
    colors = [(r, g, b) for r in rgb_values for g in rgb_values \
                for b in rgb_values]

    # Select the first num_colors colors
    colors = colors[:num_colors]

    return colors

def gen_nx_graph(node_file:str, edge_file:str) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Generate a networkx graph from node and edge files
    """
    node_df = pd.read_csv(node_file)
    edge_df = pd.read_csv(edge_file)

    node_df['isSource'] = node_df['Name'].isin(edge_df['Source'])
    node_df['isSink'] = node_df['Name'].isin(edge_df['Sink'])
    node_df['isNode'] = node_df['isSource'] | node_df['isSink']

    G = nx.from_pandas_edgelist(edge_df, 'Source', 'Sink', 'Weight')

    return G, node_df

def gen_ig_graph(node_file:str, edge_file:str) -> Tuple[ig.Graph, pd.DataFrame]:
    """
    Generate a igraph from edge file and node file
    """
    edge_df = pd.read_csv(edge_file)
    node_df = pd.read_csv(node_file)
    tuples = [tuple(x) for x in edge_df[['Source', 'Sink', 'Weight']].values]
    G = ig.Graph.TupleList(tuples, directed=True, edge_attrs=['weight'])
    return G, node_df

def gen_louvain_cluster(G:nx.Graph, node_df:pd.DataFrame,
                        seed:int = SEED) -> pd.DataFrame:
    """
    Generate a dataframe of clusters from a networkx graph
    """
    partition = community_louvain.best_partition(G, weight = 'Weight',
                                                 random_state = seed)

    cluster_df = pd.DataFrame(partition.items(), columns=['Name', 'Cluster_id'])
    node_df = node_df.merge(cluster_df, on='Name', how='left')

    return node_df

def gen_leiden_cluster(G:ig.Graph, node_df:pd.DataFrame, n_iteration:int = 100,
                       seed:int = SEED) -> pd.DataFrame:
    """
    Runs Leiden clustering on the input Graph and returns a dataframe where 
    each node is assigned a cluster_id
    """
    partition = la.find_partition(G, la.RBConfigurationVertexPartition,
                                  n_iterations=n_iteration, weights='weight',
                                  seed=seed)
    cluster_map = {}
    for cluster_id, nodes in enumerate(partition):
        for node in nodes:
            cluster_map[G.vs[node]['name']] = cluster_id

    cluster_df = pd.DataFrame(cluster_map.items(), 
                                columns=['Name', 'Cluster_id'])

    node_df = node_df.merge(cluster_df, on='Name', how='left')
    return node_df

def gen_guided_leiden_cluster(G:ig.Graph, node_df:pd.DataFrame, n_iteration:int = 100,
                       seed:int = SEED) -> pd.DataFrame:
    """
    Runs Leiden clustering with an initial guide on the input Graph and returns a 
    dataframe where each node is assigned a cluster_id
    """

    # create an empty list of cluster ids for each node
    cluster_ids = []

    # createa a node_list from the graph
    name_df = pd.DataFrame(G.vs['name'], columns=['Name'])
    
    print(name_df)
    # merge name_df with node_df based on name
    name_df = name_df.merge(node_df, on='Name', how='left')
    print(name_df)

    cluster_ids = name_df['Cluster_id'].values.tolist()
    # print total nodes in the igraph
    print("Total nodes in the graph: ", len(G.vs))
    # print length of cluster_ids
    print("Total nodes in the cluster_ids: ", len(cluster_ids))
    print(cluster_ids)

    # create a list of int for cluster_ids by using the cluster_id column
    #cluster_ids = [int(x) for x in node_df['Cluster_id'].values]

    # partition = la.RBConfigurationVertexPartition(G, initial_membership=cluster_ids)
    # optimiser = la.Optimiser()
    # diff = optimiser.optimise_partition(partition, n_iterations=n_iteration)

    partition = la.find_partition(G, la.RBConfigurationVertexPartition,
                                  initial_membership = cluster_ids,
                                  n_iterations = n_iteration, weights = 'weight',
                                  seed = seed)

    cluster_map = {}
    for cluster_id, nodes in enumerate(partition):
        for node in nodes:
            cluster_map[G.vs[node]['name']] = cluster_id

    cluster_df = pd.DataFrame(cluster_map.items(), 
                                columns=['Name', 'Cluster_id'])

    node_df = node_df.merge(cluster_df, on='Name', how='left')
    return node_df


def gen_seeded_placement_def(cluster_def:str, placement_def:str, 
                             cluster_instance_file:str,
                             dbu:int = 1000) -> None:
    cluster2loc:Dict[str, List[int]] = {}
    instance2cluster:Dict[str, str] = {}
    cluster2dim:Dict[str, List[int]] = {}
    
    ## Updated cluster2loc
    fp = open(cluster_def, 'r')
    lines = fp.readlines()
    fp.close()
    
    flag = False
    for line in lines:
        if line.startswith('COMPONENTS'):
            flag = not flag
        elif line.startswith('END COMPONENTS'):
            flag = not flag
            break
        
        if flag and re.match(r'^\s*\-', line):
            items = line.split()
            cluster2loc[items[2]] = [int(items[6]), int(items[7])]
    ## Update instance2cluster
    fp = open(cluster_instance_file, 'r')
    lines = fp.readlines()
    fp.close()
    
    for lines in lines:
        items = lines.strip().split(',')
        instance2cluster[items[1]] = items[0]
        cluster2dim[items[0]] = [int(float(items[2])*dbu),
                                 int(float(items[3])*dbu)]
    
    ## Update the placement def file
    output_def = cluster_def.replace('.def', '_seeded.def')
    fp = open(placement_def, 'r')
    lines = fp.readlines()
    fp.close()
    
    fp = open(output_def, 'w')
    inst_name = None
    x = 0
    y = 0
    for line in lines:
        if line.startswith('COMPONENTS'):
            flag = not flag    
        elif line.startswith('END COMPONENTS'):
            flag = not flag
        
        if flag:
            items = line.split()
            if items[0] == '-':
                inst_name = re.sub(r'\\','', items[1])
                if inst_name in instance2cluster.keys():
                    cluster_name = instance2cluster[inst_name]
                    loc = cluster2loc[cluster_name]
                    dim = cluster2dim[cluster_name]
                    x = loc[0] + dim[0]//2
                    y = loc[1] + dim[1]//2
                else:
                    x = 0
                    y = 0
            line = re.sub(r'\+\s+PLACED\s+\(\s+\d+\s+\d+\s+\)',
                           f'+ PLACED ( {x} {y} )', line)
        fp.write(line)
    fp.close()
    return

def update_cluster(node_df:pd.DataFrame) -> pd.DataFrame:
    max_cluster_id = node_df['Cluster_id'].max()

    if type(max_cluster_id) == float:
        max_cluster_id = int(max_cluster_id)
    elif type(max_cluster_id) == int:
        pass
    else:
        print("Error: Cluster_id is not a float")
        exit()

    color_df = pd.DataFrame()
    if node_df['Cluster_id'].isna().any():
        node_df['Cluster_id'].fillna(max_cluster_id + 1, inplace=True)
        node_df['Cluster_id'] = node_df['Cluster_id'].astype(int)
        colors = gen_rgb_color(max_cluster_id + 2)
        color_df['Cluster_id'] = range(max_cluster_id + 2)
    else:
        colors = gen_rgb_color(max_cluster_id + 1)
        color_df['Cluster_id'] = range(max_cluster_id + 1)
    
    color_df['color'] = colors
    
    node_df = node_df.merge(color_df, on='Cluster_id', how='left')
    return node_df

def run_louvain(node_file:str, edge_file:str, cluster_file:str,
                isCsv:bool = True) -> None:
    """
    Run Louvain algorithm on a networkx graph
    """
    G, node_df = gen_nx_graph(node_file, edge_file)
    clustered_df = gen_louvain_cluster(G, node_df)
    clustered_df = update_cluster(clustered_df)
    
    write_cluster_file(clustered_df, cluster_file, isCsv)
    return

def gen_sub_louvain_cluster(G:nx.Graph, node_df:pd.DataFrame,
                            cluster_df:pd.DataFrame, cluster_id:int,
                            seed:int = SEED) -> pd.DataFrame:
    sub_cluster_df = cluster_df[cluster_df['Cluster_id'] == cluster_id]
    nodes = sub_cluster_df['Name'].values.tolist()
    H = G.subgraph(nodes)

    tmp_node_df = sub_cluster_df.merge(node_df, on=node_df.columns.to_list(),\
            how='inner')
    tmp_node_df = tmp_node_df[node_df.columns.tolist()]
    tmp_node_df = gen_louvain_cluster(H, tmp_node_df, seed)
    sub_cluster_df = update_cluster(tmp_node_df)
    return sub_cluster_df

def gen_sub_leiden_cluster(G:ig.Graph, node_df:pd.DataFrame, 
                           cluster_df:pd.DataFrame, cluster_id:int,
                           n_iteration:int = 100, 
                           seed:int = SEED) -> pd.DataFrame:
    sub_cluster_df = cluster_df[cluster_df['Cluster_id'] == cluster_id]
    nodes = sub_cluster_df['Name'].values.tolist()
    indices = [G.vs.find(name).index for name in nodes]
    H = G.subgraph(indices)
    tmp_node_df = sub_cluster_df.merge(node_df, on=node_df.columns.to_list(),
                                       how='inner')
    tmp_node_df = gen_leiden_cluster(H, tmp_node_df, n_iteration, seed)
    sub_cluster_df = update_cluster(tmp_node_df)
    return sub_cluster_df

def write_cluster_file(cluster_df:pd.DataFrame, cluster_file:str,
                       isCsv:bool=False) -> None:
    fp = open(cluster_file, 'w')
    for node, cluster_id, color in \
            cluster_df[['Name', 'Cluster_id', 'color']].values.tolist():
        hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
        fp.write(f'{node} {cluster_id} {hex_color}\n')
    fp.close()
    
    if isCsv and cluster_file.endswith('.rpt'):
        cluster_df.to_csv(cluster_file.replace('.rpt', '.csv'), index=False)
    return

def run_leiden(node_file:str, edge_file:str, cluster_file:str,
               n_iteration:int = 100, seed:int = SEED, 
               isCsv:bool = True) -> None:
    """
    Generates igraph then runs leiden and writes out the cluster file
    """
    G, node_df = gen_ig_graph(node_file, edge_file)
    node_df = gen_leiden_cluster(G, node_df, n_iteration, seed)
    node_df = update_cluster(node_df)
    write_cluster_file(node_df, cluster_file, isCsv)
    return

def run_leiden_with_hint(node_file:str, edge_file:str, hint_file:str, cluster_file:str,
               n_iteration:int = 100, seed:int = SEED, 
               isCsv:bool = True) -> None:
    """
    Generates igraph then runs leiden and writes out the cluster file
    """
    G, node_df = gen_ig_graph(node_file, edge_file)
    cluster_df = pd.read_csv(hint_file)
    # get name and cluster id from cluster_df and merge with node_df
    cluster_df = cluster_df[['Name', 'Cluster_id']]
    node_df = node_df.merge(cluster_df, on='Name', how='left')
    # print headers of node_df
    print(node_df.head())

    # generate partition from leiden using the hint
    node_df = gen_guided_leiden_cluster(G, node_df, n_iteration, seed)
    print("Finished running leiden")

    node_df = update_cluster(node_df)
    write_cluster_file(node_df, cluster_file, isCsv)
    return

def student_t_distribution_metric(node_metrics:List[List[float]], node_cluster:List[int]) -> float:
    """
    Calculate the student t distribution metric for clusters
    """
    for i in range(len(node_metrics)):
        node_metrics[i] = np.array(node_metrics[i])
    node_metrics = np.array(node_metrics)
    n_clusters = len(set(node_cluster))
    n_nodes = len(node_cluster)
    cluster_centers = np.zeros((n_clusters, node_metrics.shape[1]))
    print(cluster_centers.shape)
    cluster_ids = set(node_cluster)
    # find cluster centers of each cluster
    for cluster_id in cluster_ids:
        print(cluster_id)
        print(node_cluster == cluster_id)
        print(node_metrics[node_cluster == cluster_id])
        print(np.mean(node_metrics[node_cluster == cluster_id], axis=0))
        cluster_centers[cluster_id] = np.mean(node_metrics[node_cluster == cluster_id], axis=0)
    
    # distance from each node to each cluster center
    dist = np.zeros((n_nodes, n_clusters))

    for i in range(n_nodes):
        for j in range(n_clusters):
            dist[i, j] = np.linalg.norm(node_metrics[i] - cluster_centers[j])

    # stdev of distances from each node to its cluster center
    stdev = np.std(dist, axis=1)
    # average stdev of distances from each node to its cluster center
    avg_stdev = np.mean(stdev)
    # stdev of distances from each node to other cluster centers
    stdev_other = np.zeros(n_nodes)
    for i in range(n_nodes):
        stdev_other[i] = np.std(dist[i, np.arange(n_clusters) != node_cluster[i]])

    # average stdev of distances from each node to other cluster centers
    avg_stdev_other = np.mean(stdev_other)
    # student t distribution metric
    t_dist = avg_stdev / avg_stdev_other
    return t_dist
    

def report_metrics(placement_file:str, cluster_file:str) -> None:
    placement_df = pd.read_csv(placement_file)
    cluster_df = pd.read_csv(cluster_file)
    
    df = placement_df.merge(cluster_df, on=['Name', 'Type', 'Master'],
                            how='inner')
    
    node_cluster = df['Cluster_id'].values.tolist()
    node_metrics = df[['X', 'Y']].values.tolist()
    
    dbi_score = davies_bouldin_score(node_metrics, node_cluster)
    sc_score = silhouette_score(node_metrics, node_cluster)
    vrc_score = calinski_harabasz_score(node_metrics, node_cluster)
    t_student_metric = student_t_distribution_metric(node_metrics, node_cluster)
    
    # vrc_score = report_vrc(df)
    
    print('The Clustering metrics are:')
    print(f'DBI: {dbi_score}')
    print(f'SC: {sc_score}')
    print(f'VRC: {vrc_score}')
    return

def generate_hgr_file(hgr_file:str, hgr_output_file:str) -> None:
    """
    Reads in a hgr file generated using write_hypergraph command in Innovus
    and write out the hgr file in the format required by the partitioning
    tools.
    """
    # TODO: Add code to writeout node and net mapping file
    fp = open(hgr_file, "r")
    lines = fp.readlines()
    fp.close()

    nodMape = {}
    node_id = 1
    net_count = len(lines)
    ### Update node count:
    for line in lines: 
        items = line.split()
        for item in items[1:]:
            if item not in nodMape.keys():
                nodMape[item] = node_id
                node_id += 1

    ## output hgr file
    fp = open(hgr_output_file, "w")
    fp.write(f"{net_count} {len(nodMape)}\n")
    for line in lines:
        items = line.split()
        for item in items[1:]:
            fp.write(f"{nodMape[item]} ")
        fp.write("\n")
    fp.close()

    print(f"{hgr_output_file} is generated")
    return

def gen_inst_group_def(cluster_csv:str, cluster_def:str, design:str) -> None:
    '''
    Generate a def file that create cluster groups of instances.
    cluster_csv: csv file with cluster_id, instance name and type generated by
                run_louvain or run_leiden   
    cluster_def: def file that will be generated
    '''
    if not os.path.exists(cluster_csv):
        print(f"Error: {cluster_csv} does not exist")
        exit()
    
    cluster_def_dir = os.path.dirname(cluster_def)
    if not os.path.exists(cluster_def_dir):
        print(f"Warning: {cluster_def_dir} does not exist")
        print(f"Creating {cluster_def_dir}")
        os.makedirs(cluster_def_dir)
        
    cluster_df = pd.read_csv(cluster_csv)
    cluster_df = cluster_df.sort_values(by=['Cluster_id'])
    instance_df = cluster_df[cluster_df['Type'] == 'inst']
    cluster_ids = instance_df['Cluster_id'].unique().tolist()
    
    fp = open(cluster_def, 'w')
    fp.write(f"VERSION 5.8 ;\n")
    fp.write(f"DESIGN {design} ;\n")
    fp.write(f"GROUPS {len(cluster_ids)} ;\n")
    for cluster_id in sorted(cluster_ids):
        fp.write(f"- cluster_{cluster_id} \n")
        for instance in instance_df[instance_df['Cluster_id'] ==\
                cluster_id]['Name']:
            instance = re.sub(r'^\{|\}$', '', instance)
            fp.write(f"    {instance}\n")
        fp.write(f";\n")
    
    fp.write(f"END GROUPS\n")
    fp.write(f"END DESIGN\n")
    fp.close()
    
    cluster_color = cluster_def.replace('.def', '.color')
    fp = open(cluster_color, 'w')
    for cluster_id in sorted(cluster_ids):
        color = cluster_df[cluster_df['Cluster_id'] ==\
                cluster_id]['color'].values[0]
        # print(color, type(color))
        color = tuple(map(int, re.findall(r'\d+', color)))
        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        fp.write(f"cluster_{cluster_id} {hex_color}\n")
    fp.close()
    return

def run_clustering(run_dir:str, design:str) -> None:
    
    if os.path.exists(run_dir) == False:
        print(f"Error: {run_dir} does not exist")
        exit()
    
    node_file = f"{run_dir}/{design}_nodes.csv"
    edge_file = f"{run_dir}/{design}_edges.csv"
    placement_file = f"{run_dir}/{design}_placement.csv"
    
    louvain_cluster_rpt = f"{run_dir}/{design}_cluster_louvain.rpt"
    leiden_cluster_rpt = f"{run_dir}/{design}_cluster_leiden.rpt"
    
    print('Running Louvain Clustering')
    start_time = time.time()
    run_louvain(node_file, edge_file, louvain_cluster_rpt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    
    ## Generate the cluster def file
    cluster_def = f"{run_dir}/{design}_cluster_louvain.def"
    cluster_csv = louvain_cluster_rpt.replace('.rpt', '.csv')
    gen_inst_group_def(cluster_csv, cluster_def, design)
    
    print("\nRunning Leiden Clustering")
    start_time = time.time()
    run_leiden(node_file, edge_file, leiden_cluster_rpt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    
    ## Generate the cluster def file
    cluster_def = f"{run_dir}/{design}_cluster_leiden.def"
    cluster_csv = leiden_cluster_rpt.replace('.rpt', '.csv')
    gen_inst_group_def(cluster_csv, cluster_def, design)
    
    if os.path.exists(placement_file):
        louvain_cluster_csv = louvain_cluster_rpt.replace('.rpt', '.csv')
        leiden_cluster_csv = leiden_cluster_rpt.replace('.rpt', '.csv')
        
        print("\nReporting Louvain Clustering Metrics")
        report_metrics(placement_file, louvain_cluster_csv)
        
        print("\nReporting Leiden Clustering Metrics")
        report_metrics(placement_file, leiden_cluster_csv)
    else:
        print(f"\nError: {placement_file} does not exist")

def run_clustering_with_hint(run_dir:str, design:str) -> None:
    if os.path.exists(run_dir) == False:
        print(f"Error: {run_dir} does not exist")
        exit()
    
    node_file = f"{run_dir}/{design}_nodes.csv"
    edge_file = f"{run_dir}/{design}_edges.csv"
    placement_file = f"{run_dir}/{design}_placement.csv"
    hint_file = f"{run_dir}/{design}_hint.csv"
    leiden_cluster_rpt = f"{run_dir}/{design}_cluster_leiden.rpt"
    print("\nRunning Leiden Clustering")
    start_time = time.time()
    run_leiden_with_hint(node_file, edge_file, hint_file, leiden_cluster_rpt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    
    ## Generate the cluster def file
    cluster_def = f"{run_dir}/{design}_cluster_leiden.def"
    cluster_csv = leiden_cluster_rpt.replace('.rpt', '.csv')
    gen_inst_group_def(cluster_csv, cluster_def, design)

if __name__ == '__main__':
    run_dir = sys.argv[1]
    design = sys.argv[2]
    
    # Reading a cluster file and generating def and rgb file 
    # cluster_file = "/home/fetzfs_projects/PlacementCluster/bodhi/run_ref/ariane/ariane_hier_clusters.rpt"
    # cluster_def = "/home/fetzfs_projects/PlacementCluster/bodhi/run_ref/ariane/ariane_hier_clusters.def"
    # node_file = "/home/fetzfs_projects/PlacementCluster/bodhi/run_ref/ariane/ariane_hier_nodes.csv"
    # edge_file = "/home/fetzfs_projects/PlacementCluster/bodhi/run_ref/ariane/ariane_hier_edges.csv"
    # G, node_df = gen_nx_graph(node_file, edge_file)
    # cluster_df = pd.read_csv(cluster_file)
    # # get name and cluster id from cluster_df and merge with node_df
    # cluster_df = cluster_df[['Name', 'Cluster_id']]
    # print(cluster_df.head())
    # node_df = node_df.merge(cluster_df, on='Name', how='left')
    # print(node_df.head())
    # colored_cluster_df = update_cluster(node_df)
    # print(colored_cluster_df.head())
    # write_cluster_file(colored_cluster_df, cluster_file, True)
    # cluster_colored_csv = cluster_file.replace('.rpt', '.csv')
    # gen_inst_group_def(cluster_colored_csv, cluster_def, 'ariane')
    # exit()

    # run_clustering(run_dir, design)
    # exit()

    run_clustering_with_hint(run_dir, design)
    exit()
    node_file = "/home/fetzfs_projects/PlacementCluster/sakundu/Physical"\
                "Synthesis/run_ref/jpeg_encoder_nodes.csv"
    
    edge_file = "/home/fetzfs_projects/PlacementCluster/sakundu/Physical"\
                "Synthesis/run_ref/jpeg_encoder_edges.csv"
    
    cluster_file = "/home/fetzfs_projects/PlacementCluster/sakundu/Physical"\
                "Synthesis/run_ref/jpeg_encoder_cluster_weight.rpt"
                
    cluster_leiden_file = "/home/fetzfs_projects/PlacementCluster/sakundu/"\
            "PhysicalSynthesis/run_ref/jpeg_encoder_cluster_weight_leiden.rpt"
    
    placement_file = "/home/fetzfs_projects/PlacementCluster/sakundu/"\
            "PhysicalSynthesis/run_ref/jpeg_encoder_placement.csv"
    
    louvain_cluster_csv = cluster_file.replace('.rpt', '.csv')
    leiden_cluster_csv = cluster_leiden_file.replace('.rpt', '.csv')
    
    print('Running Louvain Clustering')
    run_louvain(node_file, edge_file, cluster_file)
    report_metrics(placement_file, louvain_cluster_csv)
    
    print('\nRunning Leiden Clustering')
    run_leiden(node_file, edge_file, cluster_leiden_file)
    report_metrics(placement_file, leiden_cluster_csv)
    