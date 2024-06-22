#!/bin/bash -i
source /home/sakundu/SCRIPT/open_road_setup
module unload anaconda3
module load anaconda3/23.3.1
source $CONDA_SH
conda activate /home/tool/anaconda/envs/cluster

export run_dir=$1
export design=$2
export lef_files=$3

echo "Run dir: ${run_dir}"
echo "Design: ${design}"
echo "Lef file: ${lef_files}"

export flat_placed_def="${run_dir}/${design}_flat_global_placed.def"
blob_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement"
Scripts="${blob_dir}/"

export def_file1="${run_dir}/${design}_seeded_25_25.def"
export def_file2="${run_dir}/${design}_seeded_10_10.def"
export def_file3="${run_dir}/${design}_seeded_10_25_10_25_42.def"
export def_file4="${run_dir}/${design}_seeded_10_25_10_25_23.def"

echo "Flat placed def: ${flat_placed_def}"
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $def_file1 0.2 42
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $def_file1 0.4 42
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $def_file3 0.2 42
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $def_file3 0.4 42
conda deactivate

OR_EXE1="/home/fetzfs_projects/PlacementCluster/sakundu/OR/20240202/OpenROAD/build/src/openroad"

export suffix="25_25_0.4_42"
export def_file="${run_dir}/${design}_seeded_${suffix}.def"
mkdir -p images
$OR_EXE1 -gui ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export suffix="25_25_0.2_42"
export def_file="${run_dir}/${design}_seeded_${suffix}.def"
mkdir -p images
$OR_EXE1 -gui ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export suffix="10_25_10_25_42_0.2_42"
export def_file="${run_dir}/${design}_seeded_${suffix}.def"
mkdir -p images
$OR_EXE1 -gui ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export suffix="10_25_10_25_42_0.4_42"
export def_file="${run_dir}/${design}_seeded_${suffix}.def"
mkdir -p images
$OR_EXE1 -gui ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

