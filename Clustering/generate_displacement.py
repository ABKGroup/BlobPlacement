from __future__ import annotations
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from functools import partial
import numpy as np
import concurrent.futures
import itertools
import statistics
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional, Union, Set, TextIO
import pandas as pd
import re
import networkx as nx
import igraph as ig
import zlib
import os
import math
import time
import os
import math
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.stats.mstats import gmean

# read a def file and capture coordinates of all the cells
# return a dictionary of cell names and coordinates
# format of line with coordinates is "- fdct_zigzag/dct_mod/FE_DBTC55_n_49 C12T32_LL_IVX100 + SOURCE TIMING + PLACED ( 155176 134200 ) N
#;" where the first number is x coordinate and the second number is y coordinate

def gather_coordinates_from_def(def_file:str) -> Dict[str, Tuple[float, float]]:
    """
    Gather coordinates from a def file
    :param def: def file
    :return: a dictionary of cell names and coordinates
    """
    print(f"Reading {def_file}")
    coordinates = {}
    cell_idx = 0
    num_cells = 0
    db_unit = 1000
    is_component = False
    pattern = r'\+\s+\S+\s+\(\s+\d+\s+\d+\s+\)'
    with open(def_file, 'r') as f:
        for line in f:
            if re.match(r"^UNITS DISTANCE MICRONS", line):
                line = line.strip().split()
                db_unit = int(line[3])
            elif line.startswith("COMPONENTS"):
                line = line.strip().split()
                num_cells = int(line[1])
                is_component = True
            elif is_component and re.match(r"^\s*-", line):
                cell_idx += 1
                sub_line = re.search(pattern, line).group()
                sub_line = sub_line.strip().split()
                # print(f"Cell {cell_idx}: {sub_line}")
                line = line.strip().split()
                cell = line[1]
                cell = re.sub(r'[{}\\]', '', cell)
                x = round(float(sub_line[3])/db_unit, 6)
                y = round(float(sub_line[4])/db_unit, 6)
                coordinates[cell] = (x, y)
            elif line.startswith("END COMPONENTS"):
                is_component = False
                print(f"Total number of cells: {num_cells}, number of cells read: {cell_idx}")
                break
    return coordinates

def extract_displacement_info(csv_file:str, def1:str, def2:str, output_csv:str) -> None:
    cluster_df = pd.read_csv(csv_file, header=None)
    cluster_df.columns = ['Cluster_Id', 'Name', "Width", "Height"]
    def1_cords = gather_coordinates_from_def(def1)
    def1_cord_list = [{'Name': key, 'X1': value[0], 'Y1': value[1]} for key, value in def1_cords.items()]
    def1_df = pd.DataFrame(def1_cord_list)
    
    def2_cords = gather_coordinates_from_def(def2)
    def2_cord_list = [{'Name': key, 'X2': value[0], 'Y2': value[1]} for key, value in def2_cords.items()]
    def2_df = pd.DataFrame(def2_cord_list)
    
    df = pd.merge(cluster_df, def1_df, on='Name')
    df = pd.merge(df, def2_df, on='Name')
    df.to_csv(output_csv, index=False)
    

def extract_seeded_place_info(run_dir:str, design:str) -> None:
    '''
    Reads the <design>_cluster.csv, <design>_cluster_placed_seeded.def,
    <design>_seeded_global_placed.def and See the cell movement from the
    seed to the final placement
    '''
    csv_file = f"{run_dir}/{design}_cluster.csv"
    seeded_placed_def = f"{run_dir}/{design}_cluster_placed_seeded.def"
    global_placed_def = f"{run_dir}/{design}_seeded_global_placed.def"
    output_csv = f"{run_dir}/{design}_seeded2seeded_global_displacement_info.csv"
    output_cluster_csv = f"{run_dir}/{design}_seeded2seeded_global_displacement_info_cluster.csv"
    output_gif = f"{run_dir}/{design}_seeded2seeded_global_displacement.gif"
    output_dis_hist = f"{run_dir}/{design}_seeded2seeded_global_displacement_hist.png"
    extract_displacement_info(csv_file, seeded_placed_def, global_placed_def, output_csv)
    extract_cluster_displacement_info(output_csv, output_cluster_csv)
    generate_displacement_gif(output_csv, output_cluster_csv, output_gif)
    generate_displacement_histogram(output_csv, global_placed_def, output_dis_hist)
    return

def extract_seeded_place_info1(run_dir:str, design:str) -> None:
    '''
    Reads the <design>_cluster.csv, <design>_cluster_placed_seeded.def,
    <design>_seeded_global_placed.def and See the cell movement from the
    seed to the final placement
    '''
    csv_file = f"{run_dir}/{design}_cluster.csv"
    seeded_placed_def = f"{run_dir}/{design}_cluster_placed_seeded.def"
    global_placed_def = f"{run_dir}/{design}_flat_global_placed.def"
    output_csv = f"{run_dir}/{design}_seeded2flat_displacement_info.csv"
    output_cluster_csv = f"{run_dir}/{design}_seeded2flat_displacement_info_cluster.csv"
    output_gif = f"{run_dir}/{design}_seeded2flat_displacement_info.gif"
    output_dis_hist = f"{run_dir}/{design}_seeded2flat_displacement_hist.png"
    extract_displacement_info(csv_file, seeded_placed_def, global_placed_def, output_csv)
    extract_cluster_displacement_info(output_csv, output_cluster_csv)
    generate_displacement_gif(output_csv, output_cluster_csv, output_gif)
    generate_displacement_histogram(output_csv, global_placed_def, output_dis_hist)
    return

def extract_seeded_place_info2(run_dir:str, design:str) -> None:
    '''
    Reads the <design>_cluster.csv, <design>_cluster_placed_seeded.def,
    <design>_seeded_global_placed.def and See the cell movement from the
    seed to the final placement
    '''
    csv_file = f"{run_dir}/{design}_cluster.csv"
    global_placed_def = f"{run_dir}/{design}_seeded_global_placed.def"
    seeded_placed_def = f"{run_dir}/{design}_flat_global_placed.def"
    output_csv = f"{run_dir}/{design}_seeded_global2flat_displacement_info.csv"
    output_cluster_csv = f"{run_dir}/{design}_seeded_global2flat_displacement_info_cluster.csv"
    output_gif = f"{run_dir}/{design}_seeded_global2flat_displacement_info.gif"
    output_dis_hist = f"{run_dir}/{design}_seeded_global2flat_displacement_hist.png"
    extract_displacement_info(csv_file, global_placed_def, seeded_placed_def, output_csv)
    extract_cluster_displacement_info(output_csv, output_cluster_csv)
    generate_displacement_gif(output_csv, output_cluster_csv, output_gif)
    generate_displacement_histogram(output_csv, global_placed_def, output_dis_hist)
    return

# main function to calculate displacement
# return a dictionary of cell names and displacement
# we read to def files and calculate displacement between them for each cell

def extract_cluster_info(df, cluster_Id) -> Tuple[float, float, float, int]:
    '''
    Find average displacement, max displacement and standard deviation of
    displacement
    '''
    df = df[df['Cluster_Id'] == cluster_Id]
    average_displacement = df['Displacement'].mean()
    max_displacement = df['Displacement'].max()
    std_displacement = df['Displacement'].std()
    num_elements = df.shape[0]
    return average_displacement, max_displacement, std_displacement, num_elements

def extract_cluster_displacement_info(csv_file, outpu_csv) -> None:
    '''
    Read the csv file generate cluster wise displacement gif
    '''

    ## First extract average displacement, max displacement and standard 
    ## deviation of displacement for each cluster 
    df = pd.read_csv(csv_file)
    df['Displacement'] = np.sqrt((df['X2'] - df['X1'])**2 + (df['Y2'] - df['Y1'])**2)
    print(f"Averaging displacement: {df['Displacement'].mean()}")
    # Print the geometric mean of the displacement
    print(f"Geometric mean of displacement: {gmean(df['Displacement'])}")
    unique_clusters = df['Cluster_Id'].unique().tolist()
    cluster_details = []
    for cluster_id in tqdm(unique_clusters):
        cluster_details.append([cluster_id])
        avg_d, max_d, std_d, num_d = extract_cluster_info(df, cluster_id)
        cluster_details[-1].extend([avg_d, max_d, std_d, num_d])

    cluster_details_df = pd.DataFrame(cluster_details, columns=['Cluster_Id',
                                    'Avg_Displacement', 'Max_Displacement',
                                    'Std_Displacement', 'Num_Elements'])

    # Sort the cluster_details_df based on the average displacement
    cluster_details_df = cluster_details_df.sort_values(by=['Avg_Displacement'], ascending=False)
    cluster_details_df.to_csv(outpu_csv, index=False)
    return

## Find bbox size
def read_def_extract_bbox(def_file:str) -> Tuple[float, float]:
    # Check if the def file exists or not
    if not os.path.exists(def_file):
        print(f"Error: {def_file} does not exist")
        return 0, 0
    
    fp = open(def_file, 'r')
    db_micron = 1000.0
    width = 0.0
    height = 0.0
    for line in fp.readlines():
        if line.startswith('UNITS DISTANCE MICRONS'):
            db_micron = float(line.split()[3])
        elif line.startswith('DIEAREA'):
            items = line.split()
            bbox = [items[2], items[3], items[6], items[7]]
            bbox = [float(x)/db_micron for x in bbox]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            break
    fp.close()
    if width == 0.0 and height == 0.0:
        print(f"Error: Could not find DIEAREA in {def_file}")
        return 0, 0
    return width, height

def generate_displacement_histogram(displacement_csv:str, def_file:str,
                                    output_png:str) -> None:
    ## Ensure that the input files exist
    if not os.path.exists(displacement_csv):
        print(f"Error: {displacement_csv} does not exist")
        return
    if not os.path.exists(def_file):
        print(f"Error: {def_file} does not exist")
        return
    
    ## Read the displacement csv file
    df = pd.read_csv(displacement_csv)
    df['Displacement'] = np.sqrt((df['X2'] - df['X1'])**2 + (df['Y2'] - df['Y1'])**2)
    
    ## Read the def file and extract the bbox size
    width, height = read_def_extract_bbox(def_file)
    if width == 0.0 and height == 0.0:
        print(f"Error: Could not find DIEAREA in {def_file}")
        return
    diagonal = np.sqrt(width**2 + height**2)
    df['norm_disp'] = df['Displacement']/diagonal
    
    ## Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['norm_disp'], bins=100)
    ax.set_xlabel('Normalized Displacement')
    ax.set_ylabel('Number of Cells')
    ax.set_title(f"Normalized Displacement Histogram (Avg. Displacement = {df['norm_disp'].mean():.2f})")
    
    ## Save the histogram
    fig.savefig(output_png, bbox_inches='tight')
    return
    

# Function to update the plot for each Cluster_ID
def update_frame(cluster, df, ax, x1, y1, x2, y2):
    ax.clear()
    cluster_data = df[df['Cluster_Id'] == cluster]
    avg_displacement = cluster_data['Displacement'].mean()
    ax.quiver(cluster_data['X1'], cluster_data['Y1'], 
              cluster_data['X2'] - cluster_data['X1'], 
              cluster_data['Y2'] - cluster_data['Y1'], 
              angles='xy', scale_units='xy', scale=1, color='b')
    
    ax.scatter(cluster_data['X1'], cluster_data['Y1'], color='r', label='Initial Location')
    ax.scatter(cluster_data['X2'], cluster_data['Y2'], color='g', label='Final Location')
    ax.set_title(f"Displacements for Cluster {cluster} (Avg. Displacement = {avg_displacement:.2f})")
    ax.set_xlim([x1, x2])
    ax.set_ylim([y1, y2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)

def generate_displacement_gif(displacement_csv:str, cluster_info_csv:str,
                              output_gif:str) -> None:
    dis_df = pd.read_csv(displacement_csv)
    cluster_df = pd.read_csv(cluster_info_csv)
    file_name = output_gif.replace('.gif', '')
    
    # Sort the cluster_df based on the average displacement
    cluster_df = cluster_df.sort_values(by=['Avg_Displacement'], ascending=False)
    
    # Remove the points where average displacement is less than 1.0
    cluster_df = cluster_df[cluster_df['Avg_Displacement'] > 5.0]
    
    # Select top 10 clusters with highest average displacement
    top_clusters = cluster_df['Cluster_Id'].tolist()[:10]
    
    # Select last 10 clusters with lowest average displacement
    bottom_clusters = cluster_df['Cluster_Id'].tolist()[-10:]
    
    # Get canvas range
    x_min = min([dis_df['X1'].min(), dis_df['X2'].min()])
    x_max = max([dis_df['X1'].max(), dis_df['X2'].max()])
    y_min = min([dis_df['Y1'].min(), dis_df['Y2'].min()])
    y_max = max([dis_df['Y1'].max(), dis_df['Y2'].max()])
    
    # Create the figure and axis objects
    x_len = x_max - x_min
    y_len = y_max - y_min
    yy = int(10*(y_len/x_len))
    print(f"X min: {x_min}, X max: {x_max}, Y min: {y_min}, Y max: {y_max}, 10x{yy}")
    fig, ax = plt.subplots(figsize=(10, yy))
    
    dis_df['Displacement'] = np.sqrt((dis_df['X2'] - dis_df['X1'])**2 + (dis_df['Y2'] - dis_df['Y1'])**2)
    update = partial(update_frame, df=dis_df, ax=ax, x1=x_min, y1=y_min, 
                     x2=x_max, y2=y_max)
    
    ani = FuncAnimation(fig, update, frames=top_clusters, repeat=True)
    gif_path = f"{file_name}_top10.gif"
    ani.save(gif_path, writer='imagemagick', fps=1)
    
    fig, ax = plt.subplots(figsize=(10, yy))
    update = partial(update_frame, df=dis_df, ax=ax, x1=x_min, y1=y_min,
                    x2=x_max, y2=y_max)
    ani = FuncAnimation(fig, update, frames=bottom_clusters, repeat=True)
    gif_path = f"{file_name}_bottom10.gif"
    ani.save(gif_path, writer='imagemagick', fps=1)


if __name__ == "__main__":
    run_dir = sys.argv[1]
    design = sys.argv[2]
    extract_seeded_place_info(run_dir, design)
    extract_seeded_place_info1(run_dir, design)
    extract_seeded_place_info2(run_dir, design)