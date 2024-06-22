print("Importing modules starts")
import sys
from gen_cluster_lef_def import *
import re

print("Importing modules done")

if len(sys.argv) != 8:
    print("Usage: python3 gen_cluster_lef_def_test.py <threshold> <util>"\
            " <run_dir> <design> <output_dir>")
    exit()

threshold = float(sys.argv[1])
util = float(sys.argv[2])
run_dir = sys.argv[3]
design = sys.argv[4]
ar = float(sys.argv[5])
net_threshold = float(sys.argv[6])
output_dir = sys.argv[7]
cluster_map = None

if 'CLUSTER_MAP' in os.environ:
    cluster_map=os.environ['CLUSTER_MAP']
elif os.path.exists(f"{run_dir}/{design}_cluster_map.txt"):
    cluster_map = f"{run_dir}/{design}_cluster_map.txt"
    print(f"Cluster map file: {cluster_map}")
## If argument length is 
else:
    print(f"Cluster map file: {cluster_map} does not exists")


# Print out the inputs
print("Threshold: ", threshold)
print("Util: ", util)
print("Run dir: ", run_dir)
print("Design: ", design)
print("Output dir: ", output_dir)

print("Generating cluster lef and def")

node_file = f"{run_dir}/{design}_nodes.csv"
edge_file = f"{run_dir}/{design}_edges.csv"
input_def = f"{run_dir}/{design}_placed.def"
input_macro_csv = f"{run_dir}/{design}_macro.csv"

cluster_macro = False
if not os.path.exists(input_macro_csv):
    input_macro_csv = None
    cluster_macro = True

# input_lef = f"{run_dir}/28nm_12T.lef"
input_lef = os.getenv('lef_files')
input_lef = input_lef.split(' ')[0]
hgr_file = f"{run_dir}/{design}.hgr"

# If input lef matches with 28nm_12T.lef then tech is 28nm else tech is ng45
tech = 'ng45'
if re.search('28nm_12T.lef', input_lef):
    tech = '28nm'

if tech == 'ng45':
    dbu = 2000
    pin_layer = 'metal1'
    # pin_layer = 'M1'
elif tech == '28nm':
    dbu = 1000
    pin_layer = 'M1'
else:
    dbu = 2000
    pin_layer = 'metal1'

design_db = Design(design, edge_file, node_file, dbu = dbu, cluster_type = 'Leiden', 
              input_def = input_def, hgr_file=hgr_file, input_lef = input_lef,
              output_dir = output_dir, cluster_macro = False,
              macro_csv = input_macro_csv, cluster_map = cluster_map)

## If cluster utilization is greater than 0.003 then it will break it again
## Final cluster area is decided based on the cluster utilization 0.8.
## If total cell area in the cluster is 100 then actual cluster area in the def
## is 100/0.8 = 125

print("Pin layer: ", pin_layer)

design_db(threshold, util, pin_layer, ar, net_threshold)
