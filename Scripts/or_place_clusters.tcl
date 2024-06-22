set threshold $env(threshold)
set util $env(util)
set design $env(design)

if {[info exists env(cur_iter)]} {
    set cur_iter $env(cur_iter)
    set run_dir $env(output_dir)/seed/${cur_iter} 
} else {
    set run_dir $env(output_dir)
}

read_lef "${run_dir}/${design}_cluster.lef"
read_def "${run_dir}/${design}_cluster.def"
set output_def "${run_dir}/${design}_cluster_placed.def"
set net_weight_file "${run_dir}/${design}.netweight"

set startTime [clock milliseconds]
# Add if else loop to handle different cases
# global_placement -skip_initial_place
# global_placement -bin_grid_count 256 -overflow 0.01
set place_density_lb [gpl::get_global_placement_uniform_density]
set target_density [expr $place_density_lb + (1 - $place_density_lb) * 0.5]

if {[info exists env(cur_iter)]} {
    global_placement -net_weight_file $net_weight_file -bin_grid_count 256 \
            -overflow 0.5 -density $target_density
} else {
    global_placement -net_weight_file $net_weight_file -bin_grid_count 256 \
            -overflow 0.15 -density $target_density
}

set endTime [clock milliseconds]
set runtime [expr ($endTime - $startTime)/1000]
puts "Global Placement runtime: $runtime"
write_def $output_def
save_image "${run_dir}/${design}_cluster.jpeg"
exit
