# set lefdir "/home/zf4_projects/RePlAce/modularity/Modularity-Journal/testcases/lib-28/"
# set lefs "
#     ${lefdir}/28nm_12T.lef \
#     "
set lefs $env(lef_files)
set run_dir $env(run_dir)
set design $env(design)
set output_dir $env(output_dir)

foreach lef $lefs {read_lef $lef}
read_def ${run_dir}/${design}_placed.def

set startTime [clock milliseconds]
# global_placement -skip_initial_place
set place_density_lb [gpl::get_global_placement_uniform_density]
set target_density [expr $place_density_lb + (1 - $place_density_lb) * 0.5]
global_placement -density $target_density
set endTime [clock milliseconds]
set runtime [expr ($endTime - $startTime)/1000]
puts "Global Placement runtime: $runtime"

set threshold $env(threshold)
set util $env(util)

save_image ${output_dir}/${design}_flat_global_placed.jpeg
write_def ${output_dir}/${design}_flat_global_placed.def

detailed_placement
improve_placement
write_def ${output_dir}/${design}_flat_detailed_placed.def
exit