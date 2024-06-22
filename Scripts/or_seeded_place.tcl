# set lefdir "/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/blob_runs/ariane_test1"
# set lefdir "/home/fetzfs_projects/PlacementCluster/sakundu/PhysicalSynthesis/lib-28/"
# set lefs "  
#     ${lefdir}/28nm_12T.lef \
#     "
set lefs $env(lef_files)
set threshold $env(threshold)
set util $env(util)
set design $env(design)

if {[info exists env(cur_iter)]} {
    set cur_iter $env(cur_iter)
    set run_dir $env(output_dir)/seed/${cur_iter} 
} else {
    set run_dir $env(output_dir)
}

foreach lef $lefs {read_lef $lef}
read_def ${run_dir}/${design}_cluster_placed_seeded.def

save_image "${run_dir}/${design}_cluster_seeded.jpeg"

set startTime [clock milliseconds]
## Need to tune initial density penalty
set place_density_lb [gpl::get_global_placement_uniform_density]
set target_density [expr $place_density_lb + (1.0 - $place_density_lb) * 0.5]

if {[info exists env(cur_iter)]} {
    if {[info exists env(refine_flag)]} {
        global_placement -incremental -density $target_density -init_density_penalty 0.001
    } else {
        global_placement -incremental -density $target_density -init_density_penalty 0.001 -overflow 0.5
    }
} else {
    global_placement -incremental -density $target_density -init_density_penalty 0.001
}

set endTime [clock milliseconds]
set runtime [expr ($endTime - $startTime)/1000]
puts "Global Placement runtime: $runtime"
write_def ${run_dir}/${design}_seeded_global_placed.def

save_image "${run_dir}/${design}_seeded_global_placed.jpeg"
detailed_placement
#improve_placement
save_image "${run_dir}/${design}_seeded_detail_placed.jpeg"
exit
