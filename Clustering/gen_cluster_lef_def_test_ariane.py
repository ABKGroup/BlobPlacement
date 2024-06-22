import sys
from gen_cluster_lef_def import *

run_dir = "/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/Logs/ariane1"
design = "ariane"
node_file = f"{run_dir}/{design}_nodes.csv"
edge_file = f"{run_dir}/{design}_edges.csv"
input_def = f"{run_dir}/{design}_placed.def"
input_lef = f"{run_dir}/NangateOpenCellLibrary.tech.lef"
hgr_file = f"{run_dir}/{design}.hgr"

threshold = float(sys.argv[1])
util = float(sys.argv[2])

output = f"{run_dir}/{design}_threshold_{threshold}_util_{util}"

jpeg = Design(design, edge_file, node_file, cluster_type = 'Leiden', 
              inputDef = input_def, hgrFile=hgr_file, inputLef = input_lef,
              outputDir = output)

## If cluster utilization is greater than 0.003 then it will break it again
## Final cluster area is decided based on the cluster utilization 0.8.
## If total cell area in the cluster is 100 then actual cluster area in the def
## is 100/0.8 = 125
jpeg(threshold, util)
