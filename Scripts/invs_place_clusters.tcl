set threshold $env(threshold)
set util $env(util)
set run_dir $env(output_dir)
set design $env(design)

loadLefFile "${run_dir}/${design}_cluster.lef"
loadDefFile "${run_dir}/${design}_cluster.def"
set output_def "${run_dir}/${design}_cluster_placed.def"
set net_weight_file "${run_dir}/${design}.netweight"

set startTime [clock milliseconds]
place_design -concurrent_macros
set endTime [clock milliseconds]
set runtime [expr ($endTime - $startTime)/1000]
puts "Global Placement runtime: $runtime"
defOut $output_def
# save_image "${run_dir}/${design}_cluster.jpeg"
exit