from gen_cluster import *
import sys

if len(sys.argv) != 6 and len(sys.argv) != 7:
    print("Usage: python3 gen_seeded_place.py <threshold> <util>" \
            "<run_dir> <design> <output_dir> [{cur_iter}]")
    exit()
else:
    threshold = sys.argv[1]
    util = sys.argv[2]
    run_dir = sys.argv[3]
    design = sys.argv[4]
    output_dir = sys.argv[5]
    cur_iter = 0
    if len(sys.argv) == 7:
        cur_iter = sys.argv[6]

print("Generation Seeded placement def")

## Check if the IS_REGION environ variable exists or not
is_region = False
if 'IS_REGION' in os.environ:
    is_region = True
    print("IS_REGION is set to True")


input_def = f"{run_dir}/{design}_placed.def"
def_db_unit = get_db_unit(input_def)
cluster_def = f'{output_dir}/{design}_cluster_placed.def'
cluster_csv = f'{output_dir}/{design}_cluster.csv'

if len(sys.argv) == 7:
    cluster_def = f'{output_dir}/seed/{cur_iter}/{design}_cluster_placed.def'
    cluster_csv = f'{output_dir}/seed/{cur_iter}/{design}_cluster.csv'

gen_seeded_placement_def(cluster_def, input_def, cluster_csv, def_db_unit,
                         is_region)
