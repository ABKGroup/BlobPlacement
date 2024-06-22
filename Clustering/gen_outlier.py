import sys
import csv
import math
import re
from heapq import nsmallest
from typing import Dict, List, Tuple, Set

if len(sys.argv) != 5:
    print("Usage: python3 gen_outlier.py " \
            "<run_dir> <design> <output_dir> <cur_iter>")
    exit()

run_dir = sys.argv[1]
design = sys.argv[2]
output_dir = sys.argv[3]
cur_iter = sys.argv[4]

cluster_csv = f'{output_dir}/seed/{cur_iter}/{design}_cluster.csv'
cluster_def = f'{output_dir}/seed/{cur_iter}/{design}_cluster_placed.def' 
seeded_place_def = f'{output_dir}/seed/{cur_iter}/{design}_seeded_global_placed.def'
outlier_csv = f'{output_dir}/seed/{cur_iter}/{design}_outlier.csv'

def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def gen_outlier(cluster_csv:str, cluster_def:str, 
                seeded_place_def:str, outlier_csv:str, 
                dbu:int = 1000) -> None:

    instance2cluster: Dict[str, str] = {}
    instance2loc: Dict[str, List[int]] = {}
    cluster2loc: Dict[str, List[int]] = {}
    cluster2instance: Dict[str, List[str]] = {}
    outlier2cluster: Dict[str, str] = {}

    with open(cluster_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            cluster_name, instance_name, cluster_width, cluster_height = row
            
            # Fill instance2cluster
            instance2cluster[instance_name] = cluster_name
            
            # Fill cluster2instance
            if cluster_name not in cluster2instance:
                cluster2instance[cluster_name] = []
            cluster2instance[cluster_name].append(instance_name)

    within_components_block = False
    with open(cluster_def, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("COMPONENTS"):
                within_components_block = True
                continue  # Skip the line that only starts the COMPONENTS block
            elif line.startswith("END COMPONENTS"):
                break  # Exit the loop once we reach the end of the COMPONENTS block

            if within_components_block:
                tokens = line.split()
                cluster_name = tokens[2]
                x_coord = tokens[6]
                y_coord = tokens[7]
                
                # Fill cluster2loc
                cluster2loc[cluster_name] = [int(x_coord), int(y_coord)]

    within_components_block = False
    with open(seeded_place_def, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("COMPONENTS"):
                within_components_block = True
                continue  # Skip the line that only starts the COMPONENTS block
            elif line.startswith("END COMPONENTS"):
                break  # Exit the loop once we reach the end of the COMPONENTS block

            if within_components_block:
                tokens = line.split()
                instance_name = tokens[1].replace('\\',"")
                x_coord = tokens[6]
                y_coord = tokens[7]
                
                # Only fill instance2loc if instance_name exists in instance2cluster
                if instance_name in instance2cluster:
                    instance2loc[instance_name] = [int(x_coord), int(y_coord)]

    outliers = set() 
    # Calculate distance from cluster centroid for each instance and collect the distances
    for cluster, instances in cluster2instance.items():
        cluster_loc = cluster2loc[cluster]
        distances = []
        
        for instance in instances:
            if instance in instance2loc:  # Check if the instance has a location
                instance_loc = instance2loc[instance]
                
                # Calculate Euclidean distance
                distance = math.sqrt((cluster_loc[0] - instance_loc[0]) ** 2 + (cluster_loc[1] - instance_loc[1]) ** 2)
                distances.append(distance)
                
        # Calculate standard deviation of the distances
        mean_distance = sum(distances) / len(distances)
        variance = sum((x - mean_distance) ** 2 for x in distances) / len(distances)
        std_dev = math.sqrt(variance)
        
        # Identify outliers: those that are more than 2 standard deviations away from the mean
        for instance, distance in zip(instances, distances):
            if abs(distance - mean_distance) > 2 * std_dev:
                outliers.add(instance)

    # Reassign outliers to the nearest cluster 
    for outlier in outliers:
        outlier_loc = instance2loc[outlier]
        min_distance = float('inf')  # Initialize with infinity
        nearest_cluster = None  # Initialize with None
        
        # Find the nearest cluster
        for cluster, cluster_loc in cluster2loc.items():
            distance = math.sqrt((cluster_loc[0] - outlier_loc[0]) ** 2 + (cluster_loc[1] - outlier_loc[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster
        
        # Update cluster2instance and instance2cluster
        if nearest_cluster:
            outlier2cluster[outlier] = nearest_cluster

    with open(outlier_csv, 'w') as f:
        for outlier in outliers:
            f.write(f'{outlier} {instance2cluster[outlier]} {outlier2cluster[outlier]}\n')


gen_outlier(cluster_csv, cluster_def, seeded_place_def, outlier_csv, dbu=2000)
