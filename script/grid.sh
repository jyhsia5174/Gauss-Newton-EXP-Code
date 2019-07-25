d=40
tr='ratings.dat.trva'
va='ratings.dat.te'

max_iter=500
eps=1e-1

log_path="logs/$tr.$va.$d.${max_iter}.${eps}"
mkdir -p $log_path

# Initialization read function
matlab -nodisplay -nosplash -nodesktop -r "make;exit;" 

task(){
for l in 50 25 10 5 2.5 1 5e-1 2.5e-1 1e-1 
#for l in 5e-2 2.5e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4 
#for l in 5e-5 2.5e-5 1e-5 5e-6 2.5e-6 1e-6 5e-7 2.5e-7 1e-7 
do
    lambda_U=$l
    lambda_V=$l
    log_name="$tr.$va.$d.$lambda_U.$lambda_V"
    matlab -nodisplay -nosplash -nodesktop -r "epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter};run('example.m');exit;" > $log_path/$log_name &
done
}
task
wait

ask(){
for l in 50 25 10 5 2.5 1 5e-1 2.5e-1 1e-1 
or l in 5e-2 2.5e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4 
for l in 5e-5 2.5e-5 1e-5 5e-6 2.5e-6 1e-6 5e-7 2.5e-7 1e-7 
o
   lambda_U=$l
   lambda_V=$l
   log_name="$tr.$va.$d.$lambda_U.$lambda_V"
   matlab -nodisplay -nosplash -nodesktop -r "epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter};run('example.m');exit;" > $log_path/$log_name &
one

ask
ait

task(){
#for l in 50 25 10 5 2.5 1 5e-1 2.5e-1 1e-1 
#for l in 5e-2 2.5e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4 
for l in 5e-5 2.5e-5 1e-5 5e-6 2.5e-6 1e-6 5e-7 2.5e-7 1e-7 
do
    lambda_U=$l
    lambda_V=$l
    log_name="$tr.$va.$d.$lambda_U.$lambda_V"
    matlab -nodisplay -nosplash -nodesktop -r "epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter};run('example.m');exit;" > $log_path/$log_name &
done
}
task
wait
