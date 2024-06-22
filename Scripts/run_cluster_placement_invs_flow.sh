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
export IS_REGION=1
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
module load innovus/21.1
cd ${output_dir}
innovus -64 -init ${blob_dir}/Scripts/invs_place_clusters.tcl -overwrite -log ${log_dir}/innovus_place_clusters.log
module unload innovus/21.1
cd -

conda activate /home/tool/anaconda/envs/cluster
echo "Starting the seeded placement input generation job"
python3 ${blob_dir}/Clustering/gen_seeded_palce.py ${threshold} ${util} ${run_dir} ${design} ${output_dir}

conda deactivate
echo "Starting the seeded placement job"

module load innovus/21.1
cp -rf ${blob_dir}/flows/scripts/${design} ${output_dir}/innovus_run
cd ${output_dir}/innovus_run
export SEEDED_DEF="${output_dir}/${design}_cluster_placed_seeded.def"
${blob_dir}/flows/scripts/cadence/run.sh
cd -
