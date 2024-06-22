set lefs $env(lef_files)
puts "LEF files: $lefs"
set design $env(design)
puts "Design: $design"
set input_def $env(def_file)
puts "Input DEF: $input_def"
set suffix $env(suffix)
puts "Suffix: $suffix"
set run_dir $env(run_dir)
puts "Run directory: $run_dir"
set output_def "${run_dir}/${design}_${suffix}.def"

foreach lef $lefs {read_lef $lef}
read_def $input_def

save_image "${run_dir}/${design}_${suffix}.jpeg"

set startTime [clock milliseconds]

## Need to tune initial density penalty
set place_density_lb [gpl::get_global_placement_uniform_density]
set target_density [expr $place_density_lb + (1.0 - $place_density_lb) * 0.5]

# gpl::global_placement_debug
global_placement -density $target_density -overflow 0.12

set endTime [clock milliseconds]
set runtime [expr ($endTime - $startTime)/1000]
puts "Global Placement runtime: $runtime"
write_def $output_def

save_image "${run_dir}/${design}_global_${suffix}.jpeg"
detailed_placement
save_image "${run_dir}/${design}_detail_${suffix}.jpeg"
exit
