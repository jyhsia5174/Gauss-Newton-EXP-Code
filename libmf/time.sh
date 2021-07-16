#! /bin/bash

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
          echo './run.sh [-n] [-c <num>]' >&2
          exit 1
    esac
done

lambda=(5e-1 5e-2 5e-3)
f=0
k=20
t=1000
r=0.1

log_dir="log"
mkdir -p ${log_dir}
task(){
  for l2 in ${lambda[@]}; do
      log="f_${f}_l2_${l2}_k_${k}_t_${t}_r_${r}_time_1"
      echo "timeout 2.5h ./mf-train -l2 ${l2} -f ${f} -k ${k} -t ${t} -r ${r} -p va tr > ${log_dir}/${log}"
  done
}

if ${dry_run}; then
  echo "Dry run -c ${num_proc}"
  task
else
  echo "Run with ${num_proc} cores."
  task | xargs -0 -d '\n' -P ${num_proc} -I {} sh -c "{}" &
fi
