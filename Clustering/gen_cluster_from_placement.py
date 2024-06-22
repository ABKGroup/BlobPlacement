import sys
import os
import numpy as np
import re
from typing import List, Tuple, Optional
import time

def get_die_box(def_file:str) -> Optional[List[int]]:
    """
    Reads the def file and returns the die area

    Args:
        def_file (str): Provide the def file path

    Returns:
        Optional[List[int,int,int,int]]: Returns the die as a tuple
                                          (x1, y1, x2, y2)
    """
    
    ## Ensure the def file exists
    if not os.path.exists(def_file):
        print(f"Error: {def_file} does not exist")
        return None
    
    ## Read the def file and get the die area
    with open(def_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('DIEAREA'):
                line = line.split(' ')
                return [int(line[2]), int(line[3]), int(line[6]), int(line[7])]
    return None

def generate_bins(width:int, height:int,
                  die:List[int]) -> List[List[List[int]]]:
    ## Divide the dies into bins with width and height size
    bins = []
    
    for j in range(die[0], die[2], width):
        temp_bins = []
        j_end = j + width
        if j_end > die[2]:
            j_end = die[2]
        for i in range(die[1], die[3], height):
            i_end = i + height
            if i_end > die[3]:
                i_end = die[3]
            temp_bins.append([j, i, j_end, i_end])
        bins.append(temp_bins)
    ## Bins are sorted based on the x and y coordinates
    return bins

def generate_random_bins(width:List[int], height:List[int],
                         die:List[int], seed:int = 1) -> List[List[List[int]]]:
    np.random.seed(seed)
    bins = []
    bin_heights = []
    bin_widths = []
    
    i = die[0]
    while i < die[2]:
        temp_width = np.random.choice(range(width[0], width[1]))
        i += temp_width
        if i > die[2]:
            temp_width = die[2] - i + temp_width
        bin_widths.append(temp_width)
    
    j = die[1]
    while j < die[3]:
        temp_height = np.random.choice(range(height[0], height[1]))
        j += temp_height
        if j > die[3]:
            temp_height = die[3] - j + temp_height
        bin_heights.append(temp_height)
    
    i = die[0]
    for bin_width in bin_widths:
        j = die[1]
        temp_bins = []
        for bin_height in bin_heights:
            temp_bins.append([i, j, i+bin_width, j+bin_height])
            j += bin_height
        bins.append(temp_bins)
        i += bin_width
    
    return bins

def partition_box(width:int, height:int, imbalance:float, ar:float,
                  min_area:int) -> List[int]:
    """
    Partitions the box into two based on the width, height, imbalance, aspect

    Args:
        width (int): Widht of the box which is going to be partitioned
        height (int): Height of the box which is going to be partitioned
        imbalance (float): Imbalance of partitioning in range of 0 to 1
        ar (float): Aspect ratio of the box in range of ar to 1/ar
        min_area (int): Minimum area of the box

    Returns:
        List[int]: Returns the width of the two boxes
    """
    
    ## Min width based on aspect ratio
    ar_min_width = int(np.ceil(height*ar))
    ar_max_width = int(np.floor(height/ar))
    if ar_min_width > ar_max_width:
        print(f"Error: Aspect ratio {ar}, Min Width: {ar_min_width},"
              f" Max Width: {ar_max_width}, Height: {height}")
    
    if ar_min_width > width/2 or ar_max_width < width/2:
        print(f"Error: Aspect ratio {ar} is not possible with width {width} and"
              f" height {height}")
        return [width]
    
    ## Width range based on min area:
    area_min_width = int(np.ceil(min_area/height))
    area_max_width = width - area_min_width
    
    if area_max_width < area_min_width:
        print(f"Error: Min area {min_area} is not possible with width {width} and"
              f" height {height}")
        return [width]
    
    ## Width range based on imbalance
    imbalance_min_width = int(np.ceil(width*(1-imbalance)/2))
    imbalance_max_width = int(np.floor(width*(1+imbalance)/2))
    if imbalance_min_width > imbalance_max_width:
        print(f"Error: Imbalance {imbalance}, Min Width: {imbalance_min_width},"
              f" Max Width: {imbalance_max_width}, Width: {width}")
    
    min_width = max(ar_min_width, area_min_width, imbalance_min_width)
    max_width = min(ar_max_width, area_max_width, imbalance_max_width)
    
    if min_width > max_width:
        min_width, max_width = max_width, min_width
    
    width1 = np.random.choice(range(min_width, max_width+1))
    return [width1, width - width1]
    
def partition_helper(die:List[int], min_area:int, ar:float, is_horizontal:bool,
                     imbalance:float) -> List[List[int]]:
    """
    Helper function to partition the die into two bins based on min_area, ar and
    imbalance constraints.
    
    Args:
        die (List[int]): Die area in x1, y1, x2, y2 format
        min_area (int): Minimum area of the bin
        ar (float): Aspect ratio of the bin in range of ar to 1/ar
        is_horizontal (bool): If the partition is horizontal
        imbalance (float): Imbalance of partitioning in range of 0 to 1
    
    Returns:
        List[List[int]]: List of bins with x1, y1, x2, y2 format
    """
    ## Check if the area of the partition is more than 2x min_area
    die_area = (die[2] - die[0]) * (die[3] - die[1])
    if die_area < 2 * min_area:
        return [die]
    
    width = die[3] - die[1]
    height = die[2] - die[0]
    
    if not is_horizontal:
        width = die[2] - die[0]
        height = die[3] - die[1]
     
    new_widths = partition_box(width, height, imbalance, ar, min_area)
    
    if len(new_widths) == 1:
        return [die]
    
    if not is_horizontal:
        return [[die[0], die[1], die[0] + new_widths[0], die[3]],
                [die[0] + new_widths[0], die[1], die[2], die[3]]]
    
    return [[die[0], die[1], die[2], die[1] + new_widths[0]],
            [die[0], die[1] + new_widths[0], die[2], die[3]]]

def generate_partition_bins(die:List[int], count:int, min_area:int, ar:float,
                            imbalance:float, is_horizontal:bool = True,
                            seed:int = 42) -> List[List[int]]:
    """
    Partition the die into "count" bins based on min_area, ar and imbalance
    constraints. Here count is the maximum number of the bins to be generated.

    Args:
        die (List[int]): Die area in x1, y1, x2, y2 format
        count (int): Number of bins to be generated
        min_area (int): Minimum area of the bin
        ar (float): Aspect ratio of the bin in range of ar to 1/ar
        imbalance (float): Imbalance of partitioning in range of 0 to 1
        seed (int, optional): Seed for the experiment. Defaults to 42.

    Returns:
        List[List[int]]: List of bins with x1, y1, x2, y2 format
    """
    ar = min(ar, 1/ar)
    np.random.seed(seed)
    partition_bins = [die]

    p_counts = 1
    while len(partition_bins) < count:
        new_partition_bins = []
        is_new_partition = False
        for box in partition_bins:
            if p_counts < count:
                boxes = partition_helper(box, min_area, ar, is_horizontal,
                                     imbalance)
            else:
                boxes = [box]

            if len(boxes) == 2:
                is_new_partition = True
                p_counts += 1

            for new_box in boxes:
                new_partition_bins.append(new_box)

        ## Sort new_partition_bins based on the area and decreasign order
        new_partition_bins.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]),
                                reverse=True)
        partition_bins = new_partition_bins
        if not is_new_partition:
            break
        is_horizontal = not is_horizontal
    
    ## Sort the partition bins based on the x1, y1 coordinates
    partition_bins.sort(key=lambda x: (x[0], x[1]))
    return partition_bins 

def find_bin_center(inst:List[int],
                    bins:List[List[List[int]]]) -> Tuple[int, int]:
    """
    Find the center of the bin where the instance is placed

    Args:
        inst (List[int]): Instance location in x1, y2 format
        bin (List[List[List[int]]]): List of bins with 
                                     [columns[x1, y1, x2, y2]] format

    Returns:
        Tuple[int, int]: Returns the center of the bin
    """
    ## Find the column id using binary search
    left = 0
    right = len(bins)
    while left < right:
        mid = (left + right)//2
        if bins[mid][0][0] >= inst[0] and bins[mid][0][2] < inst[0]:
            left = mid
            break
        elif bins[mid][0][2] < inst[0]:
            left = mid + 1
        else:
            right = mid

    if left == len(bins):
        print (f"Error: Could not find the bin for {inst} in {bins[-1][0]}")
    column = bins[left]
    left = 0
    right = len(column)
    while left < right:
        mid = (left + right)//2
        if column[mid][1] >= inst[1] and column[mid][3] < inst[1]:
            left = mid
            break
        elif column[mid][3] < inst[1]:
            left = mid + 1
        else:
            right = mid
    
    x = (column[left][0] + column[left][2])//2
    y = (column[left][1] + column[left][3])//2
    
    return x, y

def generate_seeded_def(input_def: str, output_def: str,
                        bins:List[List[List[int]]]) -> None:
    ## Check the input file and output dir exists
    if not os.path.exists(input_def):
        print(f"Error: {input_def} does not exist")
        return
    
    if not os.path.exists(os.path.dirname(output_def)):
        print(f"Error: Directory of {output_def} does not exist")
        return
    
    die_x1 = bins[0][0][0]
    die_y1 = bins[0][0][1]
    die_x2 = bins[-1][-1][2]
    die_y2 = bins[-1][-1][3]
    die_mx = (die_x1 + die_x2)//2
    die_my = (die_y1 + die_y2)//2
    
    output_file = open(output_def, 'w')
    fp = open(input_def, 'r')
    in_compoenet = False
    pattern = re.compile(r'\(\s*(\d+)\s+(\d+)\s*\)')
    cell_pattern = re.compile(r'\(\s*(\d+)\s+(\d+)\s*\)')
    for line in fp:
        if line.startswith('COMPONENTS'):
            in_compoenet = True
            output_file.write(line)
            continue
        elif line.startswith('END COMPONENTS'):
            in_compoenet = False
            output_file.write(line)
            continue
        
        if in_compoenet:
            if re.match(r'^\s*-', line) and not re.match(r'^.*\+\s+FIXED\s+', line):
                ## Find the matching string ( x1 y1 ) from the line using
                ## regular expression
                match = pattern.search(line)
                
                ## We do no update location of the fixed cells
                if match:
                    x1 = int(match.group(1))
                    y1 = int(match.group(2))
                    x, y = find_bin_center([x1, y1], bins)
                    line = cell_pattern.sub(f'( {x} {y} )', line)
                else:
                    if line.endswith(';\n'):
                        ## + PLACED ( die_mx die_my ) N before the ;
                        line = line[:-2] + f' + PLACED ( {die_mx} {die_my} ) N;\n'
                    else:
                        line = line + f' + PLACED ( {die_mx} {die_my} ) N\n'
        
        output_file.write(line)
    
    fp.close()
    output_file.close()
    return


def swap_cell_in_def(input_def:str, output_def:str, swap:float,
                     seed:int = 42) -> None:
    np.random.seed(seed)
    points = {}
    cells = []
    cell_pattern = re.compile(r'\(\s*(\d+)\s+(\d+)\s*\)')
    cell_line_pattern = re.compile(r'^\s*-\s*(\S+)')  # Assume cell names don't have spaces

    start_time = time.time()
    with open(input_def, 'r') as fp:
        is_component = False
        for line in fp:
            if line.startswith('COMPONENTS'):
                is_component = True
                continue
            elif line.startswith('END COMPONENTS'):
                break
            if is_component and re.match(r'^\s*-', line) and not re.match(r'^.*\+\s+FIXED\s+', line):
                match = cell_pattern.search(line)
                x1, y1 = int(match.group(1)), int(match.group(2))
                cell = cell_line_pattern.search(line).group(1)
                point = (x1, y1)
                if point not in points:
                    points[point] = len(cells)
                    cells.append([cell])
                else:
                    cells[points[point]].append(cell)
    print(f"Time to read the def file: {time.time() - start_time}")

    start_time = time.time()
    cell_count = sum(len(cell_list) for cell_list in cells)
    
    indexes1 = np.random.choice(len(cells), int(swap*cell_count), replace=True)
    indexes2 = np.random.choice(len(cells), int(swap*cell_count), replace=True)
    sub_indexes1 = [np.random.choice(len(cells[i])) for i in indexes1]
    sub_indexes2 = [np.random.choice(len(cells[i])) for i in indexes2]
    for i in range(int(swap * cell_count)):
        cell_a, cell_b = cells[indexes1[i]][sub_indexes1[i]], cells[indexes2[i]][sub_indexes2[i]]
        cells[indexes1[i]][sub_indexes1[i]], cells[indexes2[i]][sub_indexes2[i]] = cell_b, cell_a
    print(f"Time to swap the cells: {time.time() - start_time}")

    start_time = time.time()
    cell_to_point = {cell: point for point, cell_list in zip(points.keys(), cells) for cell in cell_list}
    print(f"Time to create cell to point mapping: {time.time() - start_time}")

    start_time = time.time()
    with open(output_def, 'w') as output_file, open(input_def, 'r') as fp:
        in_component = False
        for line in fp:
            if line.startswith('COMPONENTS'):
                in_component = True
            elif line.startswith('END COMPONENTS'):
                in_component = False

            if in_component and re.match(r'^\s*-', line) and not re.match(r'^.*\+\s+FIXED\s+', line):
                cell = cell_line_pattern.search(line).group(1)
                x1, y1 = cell_to_point[cell]
                line = cell_pattern.sub(f'( {x1} {y1} )', line)
            
            output_file.write(line)
    print(f"Time to write the def file: {time.time() - start_time}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 gen_cluster_from_placement.py <def_file> <output_dir> <widht> <height>")
        sys.exit(1)
    
    def_file = sys.argv[1]
    output_def = sys.argv[2]
    
    die = get_die_box(def_file)
    if die is None:
        print(f"Error: Could not find the die area in {def_file}")
        sys.exit(1)
    
    ## Generate the bins
    if len(sys.argv) == 5:
        width = int(sys.argv[3])
        height = int(sys.argv[4])
        bins = generate_bins(width, height, die)
        print(f"Die: {die} Width: {width} Height: {height}")
    elif len(sys.argv) == 8:
        width = [int(sys.argv[3]), int(sys.argv[4])]
        height = [int(sys.argv[5]), int(sys.argv[6])]
        seed = int(sys.argv[7])
        bins = generate_random_bins(width, height, die, seed = seed)
    elif len(sys.argv) == 4:
        print("Usage: python3 gen_cluster_from_placement.py <def_file> <swap> <seed>")
        swap = float(sys.argv[2])
        seed = int(sys.argv[3])
        ## Get the directory of the def_file
        def_dir = os.path.dirname(def_file)
        def_name = os.path.basename(def_file)
        def_name = def_name.replace('.def', f'_{swap}_{seed}.def')
        output_def = f"{def_dir}/{def_name}"
        swap_cell_in_def(def_file, output_def, swap, seed = seed)
        print(f"Swapped {swap} of cells in {def_file} and saved in {output_def}")
        sys.exit(0)
    else:
        print("Usage: python3 gen_cluster_from_placement.py <def_file> <output_dir> <widht> <height>")
        sys.exit(1)
    
    ## Print the bins
    # for i, b in enumerate(bins):
    #     for j, column in enumerate(b):
    #         print(f"Bin {i} Column {j}: {column}")
    
    generate_seeded_def(def_file, output_def, bins)
    
## TODO: Start the following experiment
"""
1. Run swap experiment for jpeg untill result degrades
2. Run swap experiment for jpeg, ariane, black_parrot, mempool, megaboom
    For swap values of 0.2, 0.4, 0.8, 1.6 x (4 seeded defs)
    Total 80 runs
"""