#!/bin/bash -i
source /home/sakundu/SCRIPT/open_road_setup
# module unload anaconda3
# module load anaconda3/23.3.1
# source $CONDA_SH
# conda activate /home/tool/anaconda/envs/cluster
# 
export threshold=$1
export util=$2
export run_dir=`readlink -f $3`
export design=$4
export fanout=$5
export ar=$6
export suffix=$7

blob_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement"
log_dir="${blob_dir}/Logs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"
export lef_files="${run_dir}/Nangate45.lef"
export output_dir="${blob_dir}/blob_runs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"
mkdir -p ${log_dir} ${output_dir}
# 
# # ## Run Clustering
# echo "Starting the clustering job"
# python3 ${blob_dir}/Clustering/gen_cluster_lef_def_test.py ${threshold} ${util} ${run_dir} ${design} ${output_dir}
# conda deactivate
# 
# echo "Starting the cluster placement job"
OR_EXE1="/home/fetzfs_projects/PlacementCluster/bodhi/OpenROAD/build/src/openroad"
# $OR_EXE1 -gui ${blob_dir}/Scripts/or_place_clusters.tcl | tee ${log_dir}/or_place_clusters.log
# 
# conda activate /home/tool/anaconda/envs/cluster
# echo "Starting the seeded placement input generation job"
# python3 ${blob_dir}/Clustering/gen_seeded_palce.py ${threshold} ${util} ${run_dir} ${design} ${output_dir}
# 
# conda deactivate
# echo "Starting the seeded placement job"
# OR_EXE2="/home/fetzfs_projects/MacroPlacement/flow3/or/OpenROAD/test_build/src/openroad"
$OR_EXE1 -gui ${blob_dir}/Scripts/or_seeded_place.tcl | tee ${log_dir}/or_place_seeded.log

## Run flat placement
# echo "Starting the flat placement job"
# $OR_EXE1 -gui ${blob_dir}/Scripts/or_flat_place.tcl | tee ${log_dir}/or_only_place_flat.log
