#! /bin/bash
log_dir='log'
task(){
  for log in ${log_dir}/sol_*
  do
      awk '{ if(NF == 10) { printf("%s", $1); for(i = 2; i <= NF; i++) printf(",%s" ,$i); printf("\n");  } }' ${log} > ${log##*/}.csv 
  done
}

task
