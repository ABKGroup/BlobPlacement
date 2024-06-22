import os
import sys
import re
import time
import networkx as nx
import igraph as ig
import leidenalg as la
from typing import Tuple, List, Dict, IO
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
    # Remove {}\\ from Name column
    node_df['Name'] = node_df['Name'].replace(to_replace=r'[{}\\]', value='', regex=True)
    edge_df = pd.read_csv(edge_file)
    # Remove {}\\ from Source and Sink column
    edge_df['Source'] = edge_df['Source'].replace(to_replace=r'[{}\\]', value='', regex=True)
    edge_df['Sink'] = edge_df['Sink'].replace(to_replace=r'[{}\\]', value='', regex=True)

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
    
    # Remove {}\\ from Name column
    node_df['Name'] = node_df['Name'].replace(to_replace=r'[{}\\]', value='', regex=True)
    edge_df = pd.read_csv(edge_file)
    # Remove {}\\ from Source and Sink column
    edge_df['Source'] = edge_df['Source'].replace(to_replace=r'[{}\\]', value='', regex=True)
    edge_df['Sink'] = edge_df['Sink'].replace(to_replace=r'[{}\\]', value='', regex=True)
    
    node_df['isSource'] = node_df['Name'].isin(edge_df['Source'])
    node_df['isSink'] = node_df['Name'].isin(edge_df['Sink'])
    node_df['isNode'] = node_df['isSource'] | node_df['isSink']
    
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
                       seed:int = SEED, is_weight:bool = True) -> pd.DataFrame:
    """
    Runs Leiden clustering on the input Graph and returns a dataframe where 
    each node is assigned a cluster_id
    """
    if is_weight:
        partition = la.find_partition(G, la.RBConfigurationVertexPartition,
                                  n_iterations=n_iteration, weights='weight',
                                  seed=seed)
    else:
        partition = la.find_partition(G, la.RBConfigurationVertexPartition,
                                  n_iterations=n_iteration, seed=seed)
        
    cluster_map = {}
    for cluster_id, nodes in enumerate(partition):
        for node in nodes:
            cluster_map[G.vs[node]['name']] = cluster_id

    cluster_df = pd.DataFrame(cluster_map.items(), 
                                columns=['Name', 'Cluster_id'])

    node_df = node_df.merge(cluster_df, on='Name', how='left')
    return node_df

def read_cluster_def(cluster_def:str) -> Dict[str, List[int]]:
    """
    Reads in a cluster def file and returns a dictionary of cluster name and 
    location
    """
    cluster2loc:Dict[str, List[int]] = {}
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
    return cluster2loc

def read_cluster_instance(cluster_instance_file:str,
                          dbu:int) -> Tuple[Dict[str, List[int]],
                                        Dict[str, str]]:
    """
    Reads in a cluster instance file and returns a dictionary of cluster name
    and dimension and a dictionary of instance name and cluster name
    """
    instance2cluster:Dict[str, str] = {}
    cluster2dim:Dict[str, List[int]] = {}
    fp = open(cluster_instance_file, 'r')
    lines = fp.readlines()
    fp.close()
    
    for lines in lines:
        items = lines.strip().split(',')
        instance2cluster[items[1]] = items[0]
        cluster2dim[items[0]] = [int(float(items[2])*dbu),
                                 int(float(items[3])*dbu)]
    
    return cluster2dim, instance2cluster

def seeded_placement_def_helper(line:str, cluster2loc:Dict[str, List[int]],
                                cluster2dim:Dict[str, List[int]],
                                instance2cluster:Dict[str, str],
                                die_middle:List[int]) -> str:
    """
    Helper function for gen_seeded_placement_def
    """
    items = line.split()
    x = die_middle[0]
    y = die_middle[1]
    if items[0] == '-':
        inst_name = re.sub(r'\\','', items[1])
        if inst_name in instance2cluster.keys():
            cluster_name = instance2cluster[inst_name]
            loc = cluster2loc[cluster_name]
            dim = cluster2dim[cluster_name]
            x = loc[0] + dim[0]//2
            y = loc[1] + dim[1]//2
        else:
            x = die_middle[0]
            y = die_middle[1]
        
        pattern = r"\+\s+(PLACED|FIXED)"
        if re.search(pattern, line):
            line = re.sub(r'\+\s+PLACED\s+\(\s+\d+\s+\d+\s+\)',
                        f'+ PLACED ( {x} {y} )', line)
        else:
            if not line.endswith(';'):
                line = line + f" + PLACED ( {x} {y} ) N"
            else:
                line = re.sub(';', f" + PLACED ( {x} {y} ) N ;", line)
    
    return line

def write_region_box(cluster2loc:Dict[str, List[int]],
                     cluster2dim:Dict[str, List[int]],
                     fp:IO) -> None:
    """
    Write out the region box for each cluster
    """
    cluster_count = len(cluster2loc.keys())
    fp.write(f"REGIONS {cluster_count} ;\n")
    for cluster_name, loc in cluster2loc.items():
        dim = cluster2dim[cluster_name]
        x = loc[0]
        y = loc[1]
        fp.write(f"- {cluster_name} ( {x} {y} ) ( {x+dim[0]} {y+dim[1]} ) ;\n")
    fp.write(f"END REGIONS\n")
    return

def write_region_elements(instance2cluster:Dict[str, str],
                          fp:IO) -> None:
    """
    Write out the region elements for each cluster
    """
    cluster2instances:Dict[str, List[str]] = {}
    for instance, cluster in instance2cluster.items():
        if cluster not in cluster2instances.keys():
            cluster2instances[cluster] = []
        cluster2instances[cluster].append(instance)

    cluster_count = len(cluster2instances.keys())
    fp.write(f"GROUPS {cluster_count} ;\n")
    for cluster_name, instances in cluster2instances.items():
        fp.write(f"- {cluster_name}\n")
        for instance in instances:
            instance = re.sub(r'\[', '\\\[', instance)
            instance = re.sub(r'\]', '\\\]', instance)
            fp.write(f"    {instance}\n")
        fp.write(f" + REGION {cluster_name} ;\n")

    fp.write(f"END GROUPS\n")

    return

def get_db_unit(def_file:str) -> int:
    '''
    Returns the dbu unit of the def file
    '''
    ## First check the dbu file exists or not
    if not os.path.exists(def_file):
        print(f"Error: {def_file} does not exist")
        exit()
    
    fp = open(def_file, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if line.startswith('UNITS DISTANCE MICRONS'):
            dbu = int(line.split()[3])
            return dbu
    
    print(f"Error: UNITS DISTANCE MICRONS is not present in {def_file}")
    exit()
    

def gen_seeded_placement_def(cluster_def:str, placement_def:str,
                             cluster_instance_file:str, dbu:int = 1000,
                             is_region:bool = False) -> None:
    cluster2loc:Dict[str, List[int]] = {}
    instance2cluster:Dict[str, str] = {}
    cluster2dim:Dict[str, List[int]] = {}
    
    cluster2loc = read_cluster_def(cluster_def)
    cluster2dim, instance2cluster = read_cluster_instance(cluster_instance_file,
                                                          dbu)
    
    ## Update the placement def file
    output_def = cluster_def.replace('.def', '_seeded.def')
    fp = open(placement_def, 'r')
    lines = fp.readlines()
    fp.close()
    
    fp = open(output_def, 'w')
    
    flag = False
    
    die_middle = [0, 0]
    
    for line in lines:
        if line.startswith('DIEAREA'):
            items = line.split()
            die_middle[0] = (int(items[2]) + int(items[6]))//2
            die_middle[1] = (int(items[3]) + int(items[7]))//2
            print(f"Cluster Middle Point: {die_middle}")
            
        elif line.startswith('COMPONENTS'):
            if is_region:
                write_region_box(cluster2loc, cluster2dim, fp)
            flag = not flag    
        elif line.startswith('END COMPONENTS'):
            fp.write(line)
            if is_region:
                write_region_elements(instance2cluster, fp)
            flag = not flag
            continue
        
        if flag:
            line = seeded_placement_def_helper(line, cluster2loc, cluster2dim,
                                                instance2cluster, die_middle)
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
               isCsv:bool = True, is_weight:bool = True) -> None:
    """
    Generates igraph then runs leiden and writes out the cluster file
    """
    G, node_df = gen_ig_graph(node_file, edge_file)
    node_df = gen_leiden_cluster(G, node_df, n_iteration, seed, is_weight)
    node_df = update_cluster(node_df)
    write_cluster_file(node_df, cluster_file, isCsv)
    return

def report_metrics(placement_file:str, cluster_file:str) -> None:
    placement_df = pd.read_csv(placement_file)
    cluster_df = pd.read_csv(cluster_file)
    
    placement_df = placement_df.replace(to_replace=r'[{}\\]', value='', regex=True)
    cluster_df = cluster_df.replace(to_replace=r'[{}\\]', value='', regex=True)

    df = placement_df.merge(cluster_df, on=['Name', 'Type', 'Master'],
                            how='inner')
    
    node_cluster = df['Cluster_id'].values.tolist()
    node_metrics = df[['X', 'Y']].values.tolist()
    
    dbi_score = davies_bouldin_score(node_metrics, node_cluster)
    sc_score = silhouette_score(node_metrics, node_cluster)
    vrc_score = calinski_harabasz_score(node_metrics, node_cluster)
    
    # vrc_score = report_vrc(df)
    
    print('The Clustering metrics are:')
    print(f'DBI: {dbi_score}')
    print(f'SC: {sc_score}')
    print(f'VRC: {vrc_score}')
    return

def generate_hgr_file(hgr_file:str, node_csv:str, hgr_output_file:str) -> None:
    """
    Reads in a hgr file generated using write_hypergraph command in Innovus
    and write out the hgr file in the format required by the partitioning
    tools.
    """
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
    hgr_config = ''
    
    if node_csv is not None:
        hgr_config = '10'
    
    fp.write(f"{net_count} {len(nodMape)} {hgr_config}\n")
    for line in lines:
        items = line.split()
        for item in items[1:]:
            fp.write(f"{nodMape[item]} ")
        fp.write("\n")
    
    # Convert the nodMap to a dataframe
    df = pd.DataFrame(nodMape.items(), columns=['Name', 'Node_id'])
    if node_csv is not None:
        node_df = pd.read_csv(node_csv)
        df = df.merge(node_df, on='Name', how='inner')
        # sort dataFrame based on Node_id
        df = df.sort_values(by=['Node_id'])
        # Add area column in dataframe and it is multiple of 
        # Height and Width and make it as integer
        df['Area'] = df['Height'] * df['Width']
        df['Area'] = df['Area'].round(6)*100
        df['Area'] = df['Area'].astype(int)
    
        # Write out the area of each node in the hgr file
        for area in df['Area'].values.tolist():
            fp.write(f"{area}\n")
    
    fp.close()

    print(f"{hgr_output_file} is generated")
    
    df.to_csv(hgr_output_file.replace('.hgr', '.nodemap'), index=False)
    return

def gen_detail_from_partition(partition_file:str, node_file:str,
                              output_dir:str) -> None:
    '''
    Read in the partition file and the node_file and generate the cluster def
    and color file for visualization
    '''
    
    ## Check if the output_dir exists or not 
    if not os.path.exists(output_dir):
        # Make the output dir
        os.makedirs(output_dir)
    
    node2cluster:Dict[int, int] = {}
    fp = open(partition_file, 'r')
    
    with open(partition_file, 'r') as fp:
        node_id = 1
        for line in fp:
            node2cluster[node_id] = int(line.strip())
            node_id += 1
    
    node_df = pd.read_csv(node_file)
    cluster_df = pd.DataFrame(node2cluster.items(), columns=['Node_id', 'Cluster_id'])
    node_df = node_df.merge(cluster_df, on='Node_id', how='left')
    node_df = update_cluster(node_df)
    
    # node_file name without the extension
    node_file_name = os.path.basename(node_file).split('.')[0]
    cluster_def = f"{output_dir}/{node_file_name}_cluster.def"
    cluster_rpt = f"{output_dir}/{node_file_name}_cluster.rpt"
    cluster_csv = f"{output_dir}/{node_file_name}_cluster.csv"
    write_cluster_file(node_df, cluster_rpt, True)
    gen_inst_group_def(cluster_csv, cluster_def, node_file_name)
    
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
    
    if 'color' not in cluster_df.columns:
        print("Warning: color column is not present in the cluster file")
        return
    
    cluster_color = cluster_def.replace('.def', '.color')
    fp = open(cluster_color, 'w')
    for cluster_id in sorted(cluster_ids):
        color = cluster_df[cluster_df['Cluster_id'] ==\
                cluster_id]['color'].values[0]
        color = tuple(map(int, re.findall(r'\d+', color)))
        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        fp.write(f"cluster_{cluster_id} {hex_color}\n")
    fp.close()
    return

def run_clustering(run_dir:str, design:str, seed:int = SEED,
                   is_weight:bool = True) -> None:
    
    if os.path.exists(run_dir) == False:
        print(f"Error: {run_dir} does not exist")
        exit()
    
    node_file = f"{run_dir}/{design}_nodes.csv"
    edge_file = f"{run_dir}/{design}_edges.csv"
    placement_file = f"{run_dir}/{design}_placement.csv"
    
    louvain_cluster_rpt = f"{run_dir}/{design}_cluster_louvain.rpt"
    leiden_cluster_rpt = f"{run_dir}/{design}_cluster_leiden.rpt"
    
    print("\nRunning Leiden Clustering")
    start_time = time.time()
    run_leiden(node_file, edge_file, leiden_cluster_rpt, seed = seed, is_weight = is_weight)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    
    ## Generate the cluster def file
    cluster_def = f"{run_dir}/{design}_cluster_leiden.def"
    cluster_csv = leiden_cluster_rpt.replace('.rpt', '.csv')
    gen_inst_group_def(cluster_csv, cluster_def, design)
    
    # print('Running Louvain Clustering')
    # start_time = time.time()
    # run_louvain(node_file, edge_file, louvain_cluster_rpt)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time}")
    # 
    # ## Generate the cluster def file
    # cluster_def = f"{run_dir}/{design}_cluster_louvain.def"
    # cluster_csv = louvain_cluster_rpt.replace('.rpt', '.csv')
    # gen_inst_group_def(cluster_csv, cluster_def, design)
    # 
    # if os.path.exists(placement_file):
    #     louvain_cluster_csv = louvain_cluster_rpt.replace('.rpt', '.csv')
    #     leiden_cluster_csv = leiden_cluster_rpt.replace('.rpt', '.csv')
    #     
    #     print("\nReporting Louvain Clustering Metrics")
    #     report_metrics(placement_file, louvain_cluster_csv)
    #     
    #     print("\nReporting Leiden Clustering Metrics")
    #     report_metrics(placement_file, leiden_cluster_csv)
    # else:
    #     print(f"\nError: {placement_file} does not exist")

if __name__ == '__main__':
    run_dir = sys.argv[1]
    design = sys.argv[2]
    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])
    else:
        seed = 42
    
    is_weight = True
    if len(sys.argv) >= 5:
        if int(sys.argv[4]) == 0:
            is_weight = False
    
    run_clustering(run_dir, design, seed, is_weight)
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
    
