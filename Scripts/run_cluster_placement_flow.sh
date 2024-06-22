#!/bin/bash -i
source /home/sakundu/SCRIPT/open_road_setup
module unload anaconda3
module load anaconda3/23.3.1
source $CONDA_SH
conda activate /home/tool/anaconda/envs/cluster

export threshold=$1
export util=$2
export run_dir=`readlink -f $3`
export design=$4
export fanout=$5
export ar=$6

suffix=`date +%H%M%S_%m%d%Y`
blob_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement"
log_dir="${blob_dir}/Logs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"

# Check if the ${run_dir}/28nm_12T.lef or ${run_dir}/Nangate45.lef and based on
# the existed file set the lef_files variable
if [ -f "${run_dir}/28nm_12T.lef" ]; then
    echo "28nm_12T.lef file exists"
    export lef_files="${run_dir}/28nm_12T.lef"
elif [ -f "${run_dir}/Nangate45.lef" ]; then
    echo "Nangate45.lef file exists"
    export lef_files="${run_dir}/Nangate45.lef"
else
    echo "No lef file exists"
    exit 1
fi

echo "Lef file is ${lef_files}"

# export lef_files="${run_dir}/28nm_12T.lef"
export output_dir="${blob_dir}/blob_runs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"
mkdir -p ${log_dir} ${output_dir}

# ## Run Clustering
echo "Starting the clustering job"
python3 ${blob_dir}/Clustering/gen_cluster_lef_def_test.py ${threshold} ${util} ${run_dir} ${design} ${ar} ${fanout} ${output_dir}
conda deactivate

echo "Starting the cluster placement job"
OR_EXE1="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/Scripts/openroad"
$OR_EXE1 -gui ${blob_dir}/Scripts/or_place_clusters.tcl | tee ${log_dir}/or_place_clusters.log

conda activate /home/tool/anaconda/envs/cluster
echo "Starting the seeded placement input generation job"
python3 ${blob_dir}/Clustering/gen_seeded_palce.py ${threshold} ${util} ${run_dir} ${design} ${output_dir}

conda deactivate
echo "Starting the seeded placement job"
# OR_EXE2="/home/fetzfs_projects/MacroPlacement/flow3/or/OpenROAD/test_build/src/openroad"
OR_EXE2="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/Scripts/openroad"
$OR_EXE2 -gui ${blob_dir}/Scripts/or_seeded_place.tcl | tee ${log_dir}/or_place_seeded.log

## Run flat placement
echo "Starting the flat placement job"
$OR_EXE2 -gui ${blob_dir}/Scripts/or_flat_place.tcl | tee ${log_dir}/or_place_flat.log
