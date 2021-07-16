#!/bin/bash
#function run(solver, enable_gpu, l2, d, t, eta, cgt)
#  % function run(solver, enable_gpu, l2, d, t)
#  % Inputs:
#  % solver: 
#  %   - 0: gauss
#  %   - 1: alscg
#  %   - 2: alscg
#  % enable_gpu: 
#  %   - 0: disable
#  %   - 1: enable
#  % l2: regularization
#  % d: embedding dimemsion
#  % t: iteration
#  % eta: cg tightness
#  % cgt: max cg iteration

# Parse arg
dry_run=false
num_proc=1
while getopts 'nhc:' o; do
    case ${o} in
        n) 
          dry_run=true
          ;;
        c)
          num_proc=${OPTARG}
          ;;
        h | *) 
          echo './alscg.sh [-n] [-c <num>]' >&2
          exit 1
    esac
done

solver=1
enable_gpu=1
lambda=(5e-1 5e-2 5e-3)
d=20
t=200
eta=0.01
cgt=200
t_limit='2.5h'

log_dir="log"
mkdir -p ${log_dir}
task(){
  for l2 in ${lambda[@]}; do
      log="sol_${solver}_gpu_${enable_gpu}_l2_${l2}_d_${d}_t_${t}_eta_${eta}_cgt_${cgt}"
      echo "timeout ${t_limit} matlab -nodisplay -nosplash -nodesktop -r \"run(${solver}, ${enable_gpu}, ${l2}, ${d}, ${t}, ${eta}, ${cgt}); exit;\" > ${log_dir}/${log}"
  done
}

if ${dry_run}; then
  echo "Dry run -c ${num_proc}"
  task
else
  echo "Run"
  task | xargs -0 -d '\n' -P ${num_proc} -I {} sh -c "{}" &
fi
