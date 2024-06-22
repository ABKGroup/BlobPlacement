#!/bin/bash -i
source /home/sakundu/SCRIPT/open_road_setup
module unload anaconda3
module load anaconda3/23.3.1
source $CONDA_SH
conda activate /home/tool/anaconda/envs/cluster

export run_dir=$1
export design=$2
export input_config=$3
export iteration=$4
export lef_files=$5

echo "Run dir: ${run_dir}"
echo "Design: ${design}"
echo "Lef file: ${lef_files}"

## If iteration 1, then generate the flat placed def
if [ $iteration -eq 1 ]; then
  export flat_placed_def="${run_dir}/${design}_${input_config}.def"
else
  export flat_placed_def="${run_dir}/${design}_${input_config}_${iteration}/${design}_${input_config}.def"
fi
tcsh /home/sakundu/SCRIPT/wait.sh $flat_placed_def
blob_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement"
Scripts="${blob_dir}/"
##  Increase iteration by 1
export iteration=$((iteration+1))

export run_dir="${run_dir}/${design}_${input_config}_${iteration}"
mkdir -p ${run_dir}
cd ${run_dir}
export def_file1="${run_dir}/${design}_seeded_25_25.def"
export def_file2="${run_dir}/${design}_seeded_10_10.def"
export def_file3="${run_dir}/${design}_seeded_10_25_10_25_42.def"
export def_file4="${run_dir}/${design}_seeded_10_25_10_25_23.def"

echo "Flat placed def: ${flat_placed_def}"
echo "python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $flat_placed_def $def_file1 25 25"
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $flat_placed_def $def_file1 25000 25000
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $flat_placed_def $def_file2 10000 10000
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $flat_placed_def $def_file3 10000 25000 10000 25000 42
python3 ${blob_dir}/Clustering/gen_cluster_from_placement.py $flat_placed_def $def_file4 10000 25000 10000 25000 23
conda deactivate

OR_EXE1="/home/fetzfs_projects/PlacementCluster/sakundu/OR/20240202/OpenROAD/build/src/openroad"

export def_file="$def_file1"
export suffix="25_25"
mkdir -p images
$OR_EXE1 ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export def_file="$def_file2"
export suffix="10_10"
mkdir -p images
$OR_EXE1 ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export def_file="$def_file3"
export suffix="10_25_10_25_42"
mkdir -p images
$OR_EXE1 ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images

export def_file="$def_file4"
export suffix="10_25_10_25_23"
mkdir -p images
$OR_EXE1 ${blob_dir}/Scripts/or_place_seed.tcl -log ${run_dir}/${design}_${suffix}.log
conda activate /home/tool/anaconda/envs/cluster
python3 ${blob_dir}/Scripts/gen_gif.py ./images ${run_dir}/${design}_${suffix}.gif
conda deactivate
rm -rf ./images
