#!/bin/bash -i
source /home/kjmin/Script/or_setup # << 
module unload anaconda # << 
module load anaconda # <<
source /tool/anaconda/install/22.10/etc/profile.d/conda.sh # <<
conda activate /home/kjmin/.conda/envs/iccad23 # <<

export threshold=$1
export util=$2
export run_dir=`readlink -f $3`
export design=$4
export fanout=$5
export ar=$6

suffix=`date +%H%M%S_%m%d%Y`
blob_dir="/home/kjmin/Cowork/ucsd/BlobPlacement" # << 
log_dir="${blob_dir}/Logs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"

# Check if the ${run_dir}/28nm_12T.lef or ${run_dir}/Nangate45.lef and based on
# the existed file set the lef_files variable
if [ -f "${run_dir}/28nm_12T.lef" ]; then
    echo "28nm_12T.lef file exists"
    export lef_files="${run_dir}/28nm_12T.lef"
elif [ -f "${run_dir}/Nangate45.lef" ]; then
    echo "Nangate45.lef file exists"
    export lef_files="${run_dir}/Nangate45.lef"
else
    echo "No lef file exists"
    exit 1
fi

echo "Lef file is ${lef_files}"

# export lef_files="${run_dir}/28nm_12T.lef"
export output_dir="${blob_dir}/blob_runs/${design}_${threshold}_${util}_${ar}_${fanout}_${suffix}"
mkdir -p ${log_dir} ${output_dir}

# Run Clustering
echo "Starting the clustering job"
python3 ${blob_dir}/Clustering/gen_cluster_lef_def_test.py ${threshold} ${util} ${run_dir} ${design} ${ar} ${fanout} ${output_dir}/seed/0
conda deactivate

# Run Refining 
OR_EXE="/home/kjmin/Cowork/ucsd/OpenROAD/build/src/openroad" # <<
max_iter=10 # << 
for ((i=0; i<${max_iter}; i++));
do
    mkdir -p ${output_dir}/seed/${i} 
    mkdir -p ${log_dir}/seed/${i}
    export cur_iter=$i
    echo "Starting the #$i cluster placement job" 
    $OR_EXE ${blob_dir}/Scripts/or_place_clusters.tcl | tee ${log_dir}/seed/${i}/or_place_clusters.log

    conda activate /home/kjmin/.conda/envs/iccad23 # <<
    echo "Starting the #$i seeded placement input generation job"
    python3 ${blob_dir}/Clustering/gen_seeded_palce.py ${threshold} ${util} ${run_dir} ${design} ${output_dir} ${i}
    conda deactivate

    echo "Starting the #$i seeded placement job" 
    $OR_EXE ${blob_dir}/Scripts/or_seeded_place.tcl | tee ${log_dir}/seed/${i}/or_place_seeded.log

    if [ "$i" -eq $((max_iter - 1)) ]; then
        break
    fi
    mkdir -p "${output_dir}/seed/$((i+1))"

    echo "Staring the #$i cluster refinement job" 
    conda activate /home/kjmin/.conda/envs/iccad23 # <<
    python3 ${blob_dir}/Clustering/gen_outlier.py ${run_dir} ${design} ${output_dir} ${i}
    python3 ${blob_dir}/Clustering/gen_refined_lef_def_test.py ${threshold} ${util} ${run_dir} ${design} ${ar} ${fanout} ${output_dir} $i ${lef_files} 
    conda deactivate
done

min_value=""
min_i=""

# Select Seed
for ((i=0; i<${max_iter}; i++)); do
    file="${log_dir}/seed/${i}/or_place_seeded.log"

    if [[ -f "$file" ]]; then
        number=$(grep 'original HPWL' "$file" | head -n 1 | awk '{print $3}')
        if [[ $number ]]; then
            if [[ -z $min_value ]] || (( $(echo "$number < $min_value" | bc -l) )); then
                min_value=$number
                min_i=$i
            fi
        fi
    fi
done

echo "Minimum value is $min_value for i=$min_i"
 
export cur_iter=${min_i}
export refine_flag="DONE"
echo "Starting the final seeded placement job" 
$OR_EXE ${blob_dir}/Scripts/or_seeded_place.tcl | tee ${log_dir}/or_place_seeded.log

