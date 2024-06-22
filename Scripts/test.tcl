read_lef /home/fetzfs_projects/OpenROAD/sakundu/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef
read_lef /home/fetzfs_projects/OpenROAD/sakundu/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef
read_lef /home/fetzfs_projects/OpenROAD/sakundu/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/fakeram45_256x16.lef
read_def /home/fetzfs_projects/OpenROAD/sakundu/OpenROAD-flow-scripts/flow/results/nangate45/ariane133/ariane_noise_0/blob_runs/ariane_cluster_placed_seeded.def
set db [ord::get_db]
set block [[$db getChip] getBlock]
foreach inst [$block getInsts] {
    set pStatus [$inst getPlacementStatus]
    set inst_master [$inst getMaster]
    set master_name [$inst_master getName]
    set name [$inst getName]
    if { [string match $pStatus "NONE"] } {
        puts "$name $master_name $pStatus"
    }
}
