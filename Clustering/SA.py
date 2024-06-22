################################################################################
# Actions: swap, move, shift
# Data structure:
#       1. Cluster: Cluster dimension, list of connected nets,
#                   list of tuple of (grid_id, occupied area)
#       2. Nets: list of clusters, bbox --> {llx, lly, urx, ury}
#       3. Grids: list of tuple of {cluster, cluster area occupied in the grid},
#                   grid utilization
#       4. Design: List of Nets, List of Clusters, List of Grids,
#                   Sorted List of Grid based on utilization
#       5. SA Cost: HPWL + alpha * (avg density of top 10% congested grid)
# Input format:
#       1. CSV file for cluster dimension
#       2. CSV file for nets
#       3. Floorplan area
#       4. Pin locations
# Initialization of placement:
#       1. Initialization such that none of the grid has overflow more than 10%
#          overflow is 1 - (grid utilization). When grid utilization is greater
#          than 1.1 we consider it as 10% overflow.
# SA Moves:
#       1. Swap: Swap two clusters
#       2. Move: Move a cluster to a new location
#       3. Shift: Move the cluster to its neighboring grid
################################################################################

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union, Set, TextIO
import pandas as pd
import math

def get_grid_bbox(grid_idx:int, grid_idy:int, grid_height:float,
                  grid_width:float) -> Tuple[float, float, float, float]:
    '''
    Considers grid id starts from 0, 0
    '''
    llx = grid_idx * grid_width
    lly = grid_idy * grid_height
    urx = llx + grid_width
    ury = lly + grid_height
    return llx, lly, urx, ury

def get_grid_pt(grid_idx:int, grid_idy:int, grid_height:float,
                grid_width:float) -> Tuple[float, float]:
    '''
    Considers grid id starts from 0, 0
    '''
    pt_x = grid_idx * grid_width + grid_width/2
    pt_y = grid_idy * grid_height + grid_height/2
    return pt_x, pt_y

def get_grid_id(pt_x:float, pt_y:float, grid_height:float,
                grid_width:float) -> Tuple[int, int]:
    grid_idx = math.floor(pt_x / grid_width)
    if grid_idx < 0:
        print("ERROR: Grid idx is less than 0")
        grid_idx = 0
    
    grid_idy = math.floor(pt_y / grid_height)
    
    if grid_idy < 0:
        print("ERROR: Grid idy is less than 0")
        grid_idy = 0
    
    return grid_idx, grid_idy

def get_overlapping_area(bbox1:Tuple[float, float, float, float],
                         bbox2:Tuple[float, float, float, float]) -> float:
    llx = max(bbox1[0], bbox2[0])
    lly = max(bbox1[1], bbox2[1])
    urx = min(bbox1[2], bbox2[2])
    ury = min(bbox1[3], bbox2[3])
    
    if llx >= urx or lly >= ury:
        return 0
    else:
        return (urx - llx) * (ury - lly)
    
class Net:
    def __init__(self, id:int, name:str, weight:float = 1.0) -> None:
        self.id:int = id
        self.name:str = name
        self.weight:float = weight
        self.clusters:Set[Cluster] = set()
        self.bbox:Optional[List[float]] = None
    
    def get_hpwl(self) -> float:
        if self.bbox is None:
            return 0
        
        hpwl = self.bbox[2] - self.bbox[0] + self.bbox[3] - self.bbox[1]
        return hpwl
    
    def update_bbox(self, bbox:List[float]) -> None:
        self.bbox = bbox
    
    def add_cluster(self, cluster:Cluster) -> None:
        self.clusters.add(cluster)
    
    def update_add_cluster_bbox(self, cluster:Cluster) -> None:
        if cluster.bbox is None:
            print("Cluster is not placed. So net bbox is not updated")
            return
        
        pt_x = round(cluster.bbox[0] + cluster.width/2, 6)
        pt_y = round(cluster.bbox[1] + cluster.height/2, 6)
        
        if self.bbox is None:
            self.bbox = [pt_x, pt_y, pt_x, pt_y]
            return
        
        if self.bbox[0] > pt_x:
            self.bbox[0] = pt_x
        elif self.bbox[2] < pt_x:
            self.bbox[2] = pt_x
        
        if self.bbox[1] > pt_y:
            self.bbox[1] = pt_y
        elif self.bbox[3] < pt_y:
            self.bbox[3] = pt_y
        
    def remove_cluster(self, cluster:Cluster) -> None:
        if cluster in self.clusters:
            self.clusters.remove(cluster)
        else:
            print("Cluster is not present in the net")
        
        self.update_remove_cluster_bbox(cluster)
    
    def get_updated_net_box(self, cluster:Cluster,
                            new_pt: Tuple[float, float],
                            old_pt: Tuple[float, float]) \
                                -> Optional[List[float]]:
        
        if cluster.bbox is None:
            print("Cluster is not placed. So net bbox is not updated")
            return
        
        if self.bbox is None:
            print("Net bbox is not updated")
            return
        
        # pt_x = round(cluster.bbox[0] + cluster.width/2, 6)
        # pt_y = round(cluster.bbox[1] + cluster.height/2, 6)
        pt_x = old_pt[0]
        pt_y = old_pt[1]
        
        # If previous location is inside the box then check if the new location
        # is inside the box then return the old bbox
        # If new location is outside the box then compute new bbox
        if pt_x > self.bbox[0] and pt_x < self.bbox[2] and \
            pt_y > self.bbox[1] and pt_y < self.bbox[3]:
                # If new location is inside the box then return the old bbox
            new_bbox = self.bbox
            if new_bbox[0] > new_pt[0]:
                new_bbox[0] = new_pt[0]
            elif new_bbox[2] < new_pt[0]:
                new_bbox[2] = new_pt[0]
            
            if new_bbox[1] > new_pt[1]:
                new_bbox[1] = new_pt[1]
            elif new_bbox[3] < new_pt[1]:
                new_bbox[3] = new_pt[1]

            return new_bbox
        
        ## If the cluster was on the left boundary and new location is more left
        ## than the previous location
        if pt_x == self.bbox[0] and new_pt[0] < pt_x and pt_y > self.bbox[1] \
            and pt_y < self.bbox[3]:
            if new_pt[1] < self.bbox[1]:
                new_bbox = [new_pt[0], new_pt[1], self.bbox[2], self.bbox[3]]
            elif new_pt[1] > self.bbox[3]:
                new_bbox = [new_pt[0], self.bbox[1], self.bbox[2], new_pt[1]]
            else:
                new_bbox = [new_pt[0], self.bbox[1], self.bbox[2], self.bbox[3]]
            
            return new_bbox
        
        ## If the cluster was on the right boundary and new location is more right
        ## than the previous location
        if pt_x == self.bbox[2] and new_pt[0] > pt_x and pt_y > self.bbox[1] \
            and pt_y < self.bbox[3]:
            if new_pt[1] < self.bbox[1]:
                new_bbox = [self.bbox[0], new_pt[1], new_pt[0], self.bbox[3]]
            elif new_pt[1] > self.bbox[3]:
                new_bbox = [self.bbox[0], self.bbox[1], new_pt[0], new_pt[1]]
            else:
                new_bbox = [self.bbox[0], self.bbox[1], new_pt[0], self.bbox[3]]
            
            return new_bbox

        ## If the cluster was on the bottom boundary and new location is more bottom
        ## than the previous location
        if pt_y == self.bbox[1] and new_pt[1] < pt_y and pt_x > self.bbox[0] \
            and pt_x < self.bbox[2]:
            if new_pt[0] < self.bbox[0]:
                new_bbox = [new_pt[0], new_pt[1], self.bbox[2], self.bbox[3]]
            elif new_pt[0] > self.bbox[2]:
                new_bbox = [self.bbox[0], new_pt[1], new_pt[0], self.bbox[3]]
            else:
                new_bbox = [self.bbox[0], new_pt[1], self.bbox[2], self.bbox[3]]
            
            return new_bbox
        
        ## If the cluster was on the top boundary and new location is more top
        ## than the previous location
        if pt_y == self.bbox[3] and new_pt[1] > pt_y and pt_x > self.bbox[0] \
            and pt_x < self.bbox[2]:
            if new_pt[0] < self.bbox[0]:
                new_bbox = [new_pt[0], self.bbox[1], self.bbox[2], new_pt[1]]
            elif new_pt[0] > self.bbox[2]:
                new_bbox = [self.bbox[0], self.bbox[1], new_pt[0], new_pt[1]]
            else:
                new_bbox = [self.bbox[0], self.bbox[1], self.bbox[2], new_pt[1]]
            
            return new_bbox
        
        ## If the previous location is on the upper right corner and the new
        ## new location is more right or more upper than the previous location
        if pt_x == self.bbox[2] and pt_y == self.bbox[3] and \
            new_pt[0] >= pt_x and new_pt[1] >= pt_y:
            new_bbox = [self.bbox[0], self.bbox[1], new_pt[0], new_pt[1]]
            return new_bbox
        
        ## If the previous location is on the upper left corner and the new
        ## new location is more left or more upper than the previous location
        if pt_x == self.bbox[0] and pt_y == self.bbox[3] and \
            new_pt[0] <= pt_x and new_pt[1] >= pt_y:
            new_bbox = [new_pt[0], self.bbox[1], self.bbox[2], new_pt[1]]
            return new_bbox
        
        ## If the previous location is on the lower right corner and the new
        ## new location is more right or more lower than the previous location
        if pt_x == self.bbox[2] and pt_y == self.bbox[1] and \
            new_pt[0] >= pt_x and new_pt[1] <= pt_y:
            new_bbox = [self.bbox[0], new_pt[1], new_pt[0], self.bbox[3]]
            return new_bbox
        
        ## If the previous location is on the lower left corner and the new
        ## new location is more left or more lower than the previous location
        if pt_x == self.bbox[0] and pt_y == self.bbox[1] and \
            new_pt[0] <= pt_x and new_pt[1] <= pt_y:
            new_bbox = [new_pt[0], new_pt[1], self.bbox[2], self.bbox[3]]
            return new_bbox
        
        # Previous location is on the boundary
        # Compute new box from scracth
        new_bbox = [new_pt[0], new_pt[1], new_pt[0], new_pt[1]]
        for temp_cluster in self.clusters:
            if cluster == temp_cluster:
                continue
            
            if temp_cluster.bbox is None:
                print("Cluster is not placed. So net bbox is not updated")
                return
            
            temp_pt_x = round(temp_cluster.bbox[0] + temp_cluster.width/2, 6)
            temp_pt_y = round(temp_cluster.bbox[1] + temp_cluster.height/2, 6)
            if new_bbox[0] > temp_pt_x:
                new_bbox[0] = temp_pt_x
            elif new_bbox[2] < temp_pt_x:
                new_bbox[2] = temp_pt_x
            
            if new_bbox[1] > temp_pt_y:
                new_bbox[1] = temp_pt_y
            elif new_bbox[3] < temp_pt_y:
                new_bbox[3] = temp_pt_y

        return new_bbox
    
    def update_remove_cluster_bbox(self, cluster:Cluster) -> None:
        
        if cluster.bbox is None:
            print("Cluster is not placed. So net bbox is not updated")
            return
        
        pt_x = round(cluster.bbox[0] + cluster.width/2, 6)
        pt_y = round(cluster.bbox[1] + cluster.height/2, 6)
        
        if self.bbox is None:
            print("Net bbox is not updated as it is None")
            return
        
        # If inside the box no update is required
        if pt_x > self.bbox[0] and pt_x < self.bbox[2] and \
            pt_y > self.bbox[1] and pt_y < self.bbox[3]:
            return
        
        # If on the boundary then compute HPWL from scratch
        if pt_x == self.bbox[0] or pt_x == self.bbox[2] or \
            pt_y == self.bbox[1] or pt_y == self.bbox[3]:
            self.bbox = None
            
            for temp_cluster in self.clusters:
                if temp_cluster == cluster:
                    continue
                self.update_add_cluster_bbox(temp_cluster)
        
        return

    def compute_bbox_from_scratch(self) -> None:
        ## Ensure all the clusters are placed
        for cluster in self.clusters:
            if cluster.bbox is None:
                print("Cluster is not placed. So net bbox is not updated")
                return
        
        self.bbox = None
        for cluster in self.clusters:
            self.update_add_cluster_bbox(cluster)
    
    def compute_bbox_from_scratch_return(self) -> Optional[List[float]]:
        ## Ensure all the clusters are placed
        bkp_box = self.bbox
        for cluster in self.clusters:
            if cluster.bbox is None:
                print("Cluster is not placed. So net bbox is not updated")
                return None
        
        self.bbox = None
        for cluster in self.clusters:
            self.update_add_cluster_bbox(cluster)
        
        bbox = self.bbox
        self.bbox = bkp_box
        return bbox

class Cluster:
    def __init__(self, id:int, height:float, width:float, name:str) -> None:
        self.id:int = id
        self.height:float = height
        self.width:float = width
        self.grid_x:Optional[int] = None
        self.grid_y:Optional[int] = None
        self.offset_x:Optional[int] = None
        self.offset_y:Optional[int] = None
        self.den_matrix:Optional[List[List[float]]] = None
        self.name:str = name
        self.nets:Set[Net] = set()
        self.is_fixed = False
        # self.grids:List[Tuple[Tuple[int, int], float]] = []
        self.bbox:Optional[Tuple[float, float, float, float]] = None
    
    def fix_cluster(self):
        self.is_fixed = True
    
    def get_location(self)->Optional[Tuple[int, int]]:
        if self.grid_x is not None and self.grid_y is not None:
            return self.grid_x, self.grid_y
        else:
            print("Cluster is not placed")
        
        return None
    
    def find_grid_offset(self, grid_height, grid_width) -> Tuple[int, int]:
        
        if self.offset_x is not None and self.offset_y is not None:
            return self.offset_x, self.offset_y
        
        x_offset = 0
        y_offset = 0
        
        if self.height > grid_height:
            height_offset = (self.height - grid_height)/2
            y_offset = math.ceil(height_offset / grid_height)
        
        if self.width > grid_width:
            width_offset = (self.width - grid_width)/2
            x_offset = math.ceil(width_offset / grid_width)
        
        self.offset_x = x_offset
        self.offset_y = y_offset
        
        return x_offset, y_offset
    
    def get_density_matrix(self, grid_height:float,
                           grid_width:float) -> List[List[float]]:
        
        if self.den_matrix is not None:
            return self.den_matrix
        
        x_offset, y_offset = self.find_grid_offset(grid_height, grid_width)
        
        grid_x = 2 * x_offset + 1
        grid_y = 2 * y_offset + 1
        den_matrix = [[0.0 for _ in range(grid_x)] for _ in range(grid_y)]
        
        grid_box_width = grid_width * grid_x
        grid_box_height = grid_height * grid_y
        cluster_vllx = (grid_box_width - self.width)/2.0
        cluster_vlly = (grid_box_height - self.height)/2.0
        cluster_vurx = cluster_vllx + self.width
        cluster_vury = cluster_vlly + self.height
        cluster_vbox = cluster_vllx, cluster_vlly, cluster_vurx, cluster_vury
        
        for i in range(grid_y):
            for j in range(grid_x):
                gird_vbox = get_grid_bbox(j, i, grid_height, grid_width)
                overlapping_area = get_overlapping_area(cluster_vbox, gird_vbox)
                den_matrix[i][j] = overlapping_area / (grid_height * grid_width)
        
        self.den_matrix = den_matrix
        return den_matrix
        
    def place(self, grid_idx, grid_idy, grid_height, grid_width):
        '''
        Place command expects that grid_idx and grid_idy is a valid locatino to place
        the cluster inside the floorplan.
        '''
        ## Remove element from the grids array
        grid_center_ptx = grid_idx * grid_width + grid_width/2
        grid_center_pty = grid_idy * grid_height + grid_height/2
        cluster_llx = round(grid_center_ptx - self.width/2, 6)
        cluster_lly = round(grid_center_pty - self.height/2, 6)
        cluster_urx = cluster_llx + self.width
        cluster_ury = cluster_lly + self.height
        cluster_box = cluster_llx, cluster_lly, cluster_urx, cluster_ury
        self.update_bbox(cluster_box)
        
        self.grid_x = grid_idx
        self.grid_y = grid_idy
        # x_offset, y_offset = self.find_grid_offset(grid_height, grid_width)
        # for i in range(grid_idx - x_offset, grid_idx + x_offset + 1):
        #     for j in range(grid_idy - y_offset, grid_idy + y_offset + 1):
        #         grid_bbox = get_grid_bbox(i, j, grid_height, grid_width)
        #         overlapping_area = get_overlapping_area(cluster_box, grid_bbox)
        #         self.grids.append(((i, j), overlapping_area))
    
    def unplace(self) -> None:
        self.bbox = None
        self.grid_x = None
        self.grid_y = None
        return
        
    def update_bbox(self, bbox:Tuple[float, float, float, float]):
        self.bbox = bbox
    
    def add_net(self, net:Net):
        self.nets.add(net)
     
class Grid:
    def __init__(self, id, grid_idx, grid_idy, grid_height, grid_width) -> None:
        self.id:int = id
        self.grid_idx:int = grid_idx
        self.grid_idy:int = grid_idy
        self.grid_height:float = grid_height
        self.grid_width:float = grid_width
        self.clusters:Optional[Dict[Cluster, float]] = None
        self.cluster_area:float = 0
        self.utilization:float = 0
    
    def add_cluster(self, cluster:Cluster, overlapping_area:float) -> None:
        if self.clusters is None:
            self.clusters = {}
        
        self.clusters[cluster] = overlapping_area
        self.cluster_area += overlapping_area
        self.utilization = self.cluster_area / (self.grid_height * self.grid_width)
        
    def remove_cluster(self, cluster:Cluster) -> None:
        if self.clusters is None:
            print("Cluster is not present in the grid")
            return
        
        if cluster in self.clusters:
            self.cluster_area -= self.clusters[cluster]
            del self.clusters[cluster]
            self.utilization = self.cluster_area / (self.grid_height * self.grid_width)
            return
        
        print("Cluster is not present in the grid")
        return

class Design:
    def __init__(self, name:str, cluster_file:str, net_file:str, box_height:float,
                 box_width:float, pin_file:str, grid_x:int,
                 grid_y:int) -> None:
        self.name:str = name
        self.cluster_file:str = cluster_file
        self.net_file:str = net_file
        self.box_height:float = box_height
        self.box_width:float = box_width
        self.pin_file:str = pin_file
        self.clusters:List[Cluster] = []
        self.cluster_map:Dict[str, int] = {}
        self.nets:List[Net] = []
        self.grid_x:int = grid_x
        self.grid_y:int = grid_y
        self.grid_height = self.box_height * 1.0 / self.grid_y
        self.grid_width = self.box_width * 1.0 / self.grid_x
        self.grids:List[List[Grid]] = []
        
        ## Indicates the maximum allowed density for each grids
        self.density_threshold = 1.2

    def sort_cluster(self):
        ## Sort the cluster based on cluster area
        self.clusters.sort(key=lambda x: x.height * x.width, reverse=True)
        
## Read the design file in the following order:
    # 1. Cluster file
    #       id, height, width, name, is_fixed
    # 2. Pin file
    #       name, x, y
    # 3. Net file
    #    net_name, weight, cluster1, cluster2, cluster3, ...
    
    def read_cluster_file(self):
        ## Cluster file is in csv format
        ## Containing the four field id,hieght,width,name,is_fixed
        cluster_df = pd.read_csv(self.cluster_file)
        for idx, row in cluster_df.iterrows():
            cluster = Cluster(int(idx), row['height'], row['width'], row['name'])
            self.cluster_map[cluster.name] = int(idx)
            if row['is_fixed'] == 1:
                cluster.fix_cluster()
            self.clusters.append(cluster)
        
        self.sort_cluster()
    
    def read_pin_files(self):
        # Pins are considered as fix clusters and does not have any area impact
        df = pd.read_csv(self.pin_file)
        cluster_id = len(self.clusters)
        for _, row in df.iterrows():
            cluster = Cluster(cluster_id, 0, 0, row['name'])
            gird_x, grid_y = get_grid_id(row['x'], row['y'], self.grid_height,
                                         self.grid_width)
            cluster.place(gird_x, grid_y, self.grid_height, self.grid_width)
            cluster.fix_cluster()
            self.clusters.append(cluster)
    
    def read_net_file(self):
        # In the net file the first column is the net name and rest of the
        # columns are the source cluster and then sink clusters
        
        fp = open(self.net_file, 'r', encoding='utf-8')
        net_id = 0
        for line in fp.readlines():
            items = line.split(',')
            net = Net(net_id, items[0])
            for item in items[1:]:
                if item not in self.cluster_map:
                    print(f"ERROR Cluster {item} is not present in the design")
                    continue
                
                cluster_id = self.cluster_map[item]
                cluster = self.clusters[cluster_id]
                net.add_cluster(cluster)
                cluster.add_net(net)

            self.nets.append(net)
            net_id += 1
        fp.close()
    
    def is_feasible_location(self, cluster:Cluster, grid_x:int,
                             grid_y:int) -> bool:
        '''
        First unplace the cluster before checking feasibility of the cluster
        placement
        '''
        den_matrix = cluster.get_density_matrix(self.grid_height,
                                                    self.grid_width)
        is_valid_grid = True
        for i, den_row in enumerate(den_matrix):
            for j, den in enumerate(den_row):
                if grid_x + j >= self.grid_x or \
                    grid_y + i >= self.grid_y:
                    is_valid_grid = False
                    break
                
                if den + \
                    self.grids[grid_y+i][grid_x+j].utilization > \
                    self.density_threshold:
                        is_valid_grid = False
                        break
        return is_valid_grid
    
    def is_feasible_location_center(self, cluster:Cluster, grid_x:int,
                                    grid_y:int) -> bool:
        offset_x, offset_y = cluster.find_grid_offset(self.grid_height,
                                                        self.grid_width)
        grid_x -= offset_x
        grid_y -= offset_y
        return self.is_feasible_location(cluster, grid_x, grid_y)
    
    def place_cluster(self, cluster:Cluster, grid_x:int, grid_y:int) -> None:
        den_matrix = cluster.get_density_matrix(self.grid_height,
                                                    self.grid_width)
        for i, den_row in enumerate(den_matrix):
            for j, den in enumerate(den_row):
                overlapping_area = den*self.grid_height*self.grid_width
                self.grids[grid_y+i][grid_x+j].add_cluster(cluster,
                                                           overlapping_area)
        
        
        offset_x, offset_y = cluster.find_grid_offset(self.grid_height,
                                                      self.grid_width)
        
        grid_x += offset_x
        grid_y += offset_y
        cluster.place(grid_x, grid_y, self.grid_height, self.grid_width)
    
    def place_cluster_center(self, cluster:Cluster, grid_x:int,
                             grid_y:int) -> None:
        offset_x, offset_y = cluster.find_grid_offset(self.grid_height,
                                                        self.grid_width)
        grid_x -= offset_x
        grid_y -= offset_y
        self.place_cluster(cluster, grid_x, grid_y)

    def remove_cluster(self, cluster:Cluster) -> None:
        offset_x, offset_y = cluster.find_grid_offset(self.grid_height,
                                                        self.grid_width)
        grid_x, grid_y = cluster.get_location()
        gird_llx = grid_x - offset_x
        gird_lly = grid_y - offset_y
        gird_urx = grid_x + offset_x + 1
        gird_ury = grid_y + offset_y + 1
        for i in range(gird_lly, gird_ury):
            for j in range(gird_llx, gird_urx):
                self.grids[i][j].remove_cluster(cluster)
        
    ## Initialize the grids
    def init_grids(self):    
        for i in range(self.grid_y):
            grid_x_list = []
            for j in range(self.grid_x):
                grid_x_list.append(Grid(i * self.grid_x + j, j, i,
                                        self.grid_height, self.grid_width))
            self.grids.append(grid_x_list)

    ## Initialize cluster placement
    def init_greedy_packing(self):
        '''
        Initialize the greedy packing
        '''
        grid_x = 0
        grid_y = 0
        
        for cluster in self.clusters:
            ## Validate if the current location is feasible or not
            found_location = False
            while grid_y < self.grid_y:
                while grid_x < self.grid_x:
                    is_valid_grid = self.is_feasible_location(cluster, grid_x,
                                                              grid_y)
                    
                    if is_valid_grid:
                        found_location = True
                        break
                    grid_x += 1
                
                if found_location:
                    break
                
                grid_x = 0
                grid_y += 1
            
            ## Place the cluster in the grid
            self.place_cluster(cluster, grid_x, grid_y) 

    def spiral_packing(self):
        '''
        Initialize the spiral packing
        '''
        grid_x = 0
        grid_y = 0
        
        dir_row = [0, 1, 0, -1]
        dir_col = [1, 0, -1, 0]
        dir_id = 0
        is_visited = [[False for _ in range(self.grid_x)] for _ in range(self.grid_y)]
        
        not_enough_space = False
        
        for cluster in self.clusters:
            ## Find a valid location to place the cluster
            found_location = False
            while not found_location:
                is_valid_grid = self.is_feasible_location(cluster, grid_x,
                                                          grid_y)
                
                if is_valid_grid:
                    found_location = True
                    break
                
                is_visited[grid_y][grid_x] = True
                next_grid_x = grid_x + dir_row[dir_id]
                next_grid_y = grid_y + dir_col[dir_id]
                
                if next_grid_x >= 0 and next_grid_x < self.grid_x and \
                    next_grid_y >= 0 and next_grid_y < self.grid_y and \
                    not is_visited[next_grid_y][next_grid_x]:
                        grid_x = next_grid_x
                        grid_y = next_grid_y
                else:
                    dir_id = (dir_id + 1) % 4
                    grid_x = grid_x + dir_row[dir_id]
                    grid_y = grid_y + dir_col[dir_id]
                
                if is_visited[grid_y][grid_x]:
                    print("ERROR: No valid location found to place the cluster")
                    not_enough_space = True
                    break
            
            if not_enough_space:
                break
                
            ## Place the cluster in the grid
            ## Place the cluster in the grid
            self.place_cluster(cluster, grid_x, grid_y)
            
        if not_enough_space:
            print("ERROR: Not enough space to place all the clusters")
            return

    def swap_cluster(self, cluster1:Cluster, cluster2:Cluster) -> bool:
        '''
        Swap location of two cluster and returns if swap complete
        successfully
        '''
        if cluster1 == cluster2:
            return True
        
        grid_x1, grid_y1 = cluster1.get_location()
        grid_x2, grid_y2 = cluster2.get_location()
        
        self.remove_cluster(cluster1)
        self.remove_cluster(cluster2)
        
        is_cluster1_feasible = self.is_feasible_location_center(cluster1,
                                                                grid_x2,
                                                                grid_y2)
        is_cluster2_feasible = False
        if is_cluster1_feasible:
            is_cluster2_feasible = self.is_feasible_location_center(cluster2,
                                                                    grid_x1,
                                                                    grid_y1)
        
        if is_cluster2_feasible:
            self.place_cluster_center(cluster1, grid_x2, grid_y2)
            self.place_cluster_center(cluster2, grid_x1, grid_y1)
            return True
        
        self.place_cluster_center(cluster1, grid_x1, grid_y1)
        self.place_cluster_center(cluster2, grid_x2, grid_y2)
        
        return False

    def swap_cluster_return_hpwl_delta(self, cluster1:Cluster,
                                       cluster2:Cluster) -> float:
        '''
        Compute HPWL of the nets associated with the cluster1 and cluster2
        '''
        pre_hpwl = 0
        uniqe_nets:set[Net] = set()
        for net in cluster1.nets:
            uniqe_nets.add(net)

        for net in cluster2.nets:
            uniqe_nets.add(net)

        for net in uniqe_nets:
            pre_hpwl += net.get_hpwl()

        # Compute HPWL of the nets associated with the cluster1 and cluster2
        self.swap_cluster(cluster1, cluster2)

        post_hpwl = 0
        for net in uniqe_nets:
            bbox = net.compute_bbox_from_scratch_return()
            if bbox is None:
                print("ERROR: Net bbox is None as some clusters are not placed")
                return 0
            net_hpwl = bbox[2] - bbox[0] + bbox[3] - bbox[1]
            post_hpwl += net_hpwl

        delta_hpwl = post_hpwl - pre_hpwl
        return delta_hpwl

    def move_cluster(self, cluster:Cluster, new_x:int, new_y:int) -> bool:
        '''
        Move cluster to new location nex_x, new_y and returns true if move is
        successful
        '''
        grid_x, grid_y = cluster.get_location()
        if grid_x == new_x and grid_y == new_y:
            return True

        self.remove_cluster(cluster)
        is_feasible = self.is_feasible_location_center(cluster, new_x, new_y)
        if is_feasible:
            self.place_cluster_center(cluster, new_x, new_y)
            return True

        self.place_cluster_center(cluster, grid_x, grid_y)

        return False

    def move_cluster_return_hpwl(self, cluster:Cluster, new_x:int,
                                 new_y:int) -> float:

        old_pt_x, old_pt_y = get_grid_pt(cluster.grid_x, cluster.grid_y,
                                         self.grid_height, self.grid_width)

        new_pt_x, new_pt_y = get_grid_pt(new_x, new_y, self.grid_height,
                                            self.grid_width)

        pre_hpwl = 0
        for net in cluster.nets:
            pre_hpwl += net.get_hpwl()

        is_move_cluster = self.move_cluster(cluster, new_x, new_y)

        if not is_move_cluster:
            return 0

        post_hpwl = 0
        for net in cluster.nets:
            bbox = net.get_updated_net_box(cluster, (old_pt_x, old_pt_y),
                                            (new_pt_x, new_pt_y))

            if bbox is None:
                print("ERROR: Net bbox is None as some clusters are not placed")
                return 0
            net_hpwl = bbox[2] - bbox[0] + bbox[3] - bbox[1]
            post_hpwl += net_hpwl

        return post_hpwl - pre_hpwl

    def shift_cluster(self, cluster:Cluster, direction:List[int]) -> bool:
        '''
        Shift the cluster to its neighbouring location if feasible
        dir: [0, 1] -> Right
        dir: [0, -1] -> Left
        dir: [1, 0] -> Up
        dir: [-1, 0] -> Down
        '''

        grid_x, gird_y = cluster.get_location()
        new_x = grid_x + direction[0]
        new_y = gird_y + direction[1]

        if new_x < 0 or new_x >= self.grid_x or \
            new_y < 0 or new_y >= self.grid_y:
            return False

        return self.move_cluster(cluster, new_x, new_y)
    
    def shift_cluster_return_hpwl(self, cluster:Cluster,
                                  direction:List[int]) -> float:
        '''
        Shift the cluster to its neighbouring location if feasible
        dir: [0, 1] -> Right
        dir: [0, -1] -> Left
        dir: [1, 0] -> Up
        dir: [-1, 0] -> Down
        '''

        grid_x, gird_y = cluster.get_location()
        new_x = grid_x + direction[0]
        new_y = gird_y + direction[1]

        if new_x < 0 or new_x >= self.grid_x or \
            new_y < 0 or new_y >= self.grid_y:
            return 0

        return self.move_cluster_return_hpwl(cluster, new_x, new_y)

class SA:
    def __init__(self, design:str, run_dir:str, cluster_file:str, pin_file:str,
                 net_file:str, design_file:str) -> None:
        
        self.design = design
        self.run_dir = run_dir
        self.cluster_file = cluster_file
        self.pin_file = pin_file
        self.net_file = net_file
        self.design_file = design_file
        self.design_obj = None
        
        ## SA Configuration and other tunable parameter for SA ##
        self.grid_count_x = 256
        self.grid_count_y = 256

        self.max_iteration = 1000000
        self.max_temperature = 1e-3
        self.min_temperature = 1e-8
        self.acceptance_tolerance = 1e-9
        self.cur_hpwl = 0

    def read_design_file(self):
        '''
        Design file contains deisgn height and widht information in the
        first line
        '''

        fp = open(self.design_file, 'r', encoding='utf-8')
        line = fp.readline()
        items = line.split(',')
        design_height = float(items[0])
        design_width = float(items[1])
        fp.close()

        self.design_obj = Design(self.design, self.cluster_file, self.net_file,
                                 design_height, design_width, self.pin_file,
                                 self.grid_count_x, self.grid_count_y)
        
        self.design_obj.read_cluster_file()
        self.design_obj.read_pin_files()
        self.design_obj.read_net_file()
        self.design_obj.init_grids()
        self.design_obj.init_greedy_packing()
        for net in self.design_obj.nets:
            net.compute_bbox_from_scratch()
            self.cur_hpwl += net.get_hpwl()
        
        def runSA():
            temp_factor = math.log(self.min_temperature / self.max_temperature)
            temp = self.max_temperature
            
            
            
            