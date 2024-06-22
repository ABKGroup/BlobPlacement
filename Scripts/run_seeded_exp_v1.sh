#!/bin/bash -i
lef_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/flows/inputs/ng45/lef/"
lef_files="${lef_dir}/NangateOpenCellLibrary.tech.lef ${lef_dir}/NangateOpenCellLibrary.macro.mod.lef"
rams=`ls ${lef_dir}/fake*.lef`
lef_files="${lef_files} $rams"
script_dir="/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/Scripts/"
cd $1
${script_dir}/run_only_seeded_placement_place_v1.sh $1 $2 $3 $4 "$lef_files"
cd -
