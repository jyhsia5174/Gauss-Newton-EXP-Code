#! /bin/bash
solver=0
enable_gpu=0
lambda=(1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3)
d=40
t=100
eta=0.3
cgt=20

log_dir='log'
task(){
  for l2 in ${lambda[@]}; do
      log="sol_${solver}_gpu_${enable_gpu}_l2_${l2}_d_${d}_t_${t}_eta_${eta}_cgt_${cgt}"
      awk '{ if(NF == 10) { printf("%s", $1); for(i = 2; i <= NF; i++) printf(",%s" ,$i); printf("\n");  } }' ${log_dir}/${log} > ${log}.csv 
  done
}

task
