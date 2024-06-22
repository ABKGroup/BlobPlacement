#!/bin/bash -i
source /home/sakundu/SCRIPT/open_road_setup
module unload anaconda3
module load anaconda3/23.3.1
source $CONDA_SH
conda activate /home/tool/anaconda/envs/cluster

export run_dir=$1
export design=$2
export suff=$3
export swap=$4
export lef_files=$5

echo "Run dir: ${run_dir}"
echo "Design: ${design}"
echo "Suffix: ${suff} Swap: ${swap}"
echo "Lef file: ${lef_files}"

export flat_placed_def="${run_dir}/${design}_flat_global_placed.def"
blob_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement"
Scripts="${blob_dir}/"

export def_file1="${run_dir}/${design}_seeded_${suff}.def"

echo "Flat placed def: ${flat_placed_def}"
echo "Def file: ${def_file1}"
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $def_file1 ${swap} 42
conda deactivate

OR_EXE1="/home/fetzfs_projects/PlacementCluster/sakundu/OR/20240202/OpenROAD/build/src/openroad"

export suffix="${suff}_${swap}_42"
export def_file="${run_dir}/${design}_seeded_${suffix}.def"
mkdir -p images
$OR_EXE1 ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
# conda activate /home/tool/anaconda/envs/cluster
# python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
# conda deactivate
# rm -rf ./images