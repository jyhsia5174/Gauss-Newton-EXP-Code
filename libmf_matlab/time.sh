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
enable_gpu=false
num_proc=1
while getopts 'nhgc:' o; do
    case ${o} in
        n) 
          dry_run=true
          ;;
        c)
          num_proc=${OPTARG}
          ;;
        g)
          echo 'Enable gpu'
          enable_gpu=true
          ;;
        h | *) 
          echo './gauss.sh [-n] [-g] [-c <num>]' >&2
          exit 1
    esac
done

if ${enable_gpu}
then
  enable_gpu=1
else
  enable_gpu=0
fi
lambda=(1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3)
d=40
t=100
eta=0.3
cgt=20

log_dir="log"
mkdir -p ${log_dir}
gauss(){
  solver=0
  for l2 in ${lambda[@]}; do
      log="sol_${solver}_gpu_${enable_gpu}_l2_${l2}_d_${d}_t_${t}_eta_${eta}_cgt_${cgt}_time_1"
      echo "timeout 15m matlab -nodisplay -nosplash -nodesktop -r \"run(${solver}, ${enable_gpu}, ${l2}, ${d}, ${t}, ${eta}, ${cgt}); exit;\" > ${log_dir}/${log}"
  done
}

alscg(){
  solver=1
  for l2 in ${lambda[@]}; do
      log="sol_${solver}_gpu_${enable_gpu}_l2_${l2}_d_${d}_t_${t}_eta_${eta}_cgt_${cgt}_time_1"
      echo "timeout 15m matlab -nodisplay -nosplash -nodesktop -r \"run(${solver}, ${enable_gpu}, ${l2}, ${d}, ${t}, ${eta}, ${cgt}); exit;\" > ${log_dir}/${log}"
  done
}

if ${dry_run}; then
  echo "Dry run -c ${num_proc}"
  gauss > task.txt
  alscg >> task.txt
  cat task.txt
  rm task.txt
else
  echo "Run"
  gauss > task.txt
  alscg >> task.txt
  cat task.txt
  cat task.txt | xargs -0 -d '\n' -P ${num_proc} -I {} sh -c "{}" &
  rm task.txt
fi

wait
