
echo "check logs permission"

d=40
tr='tr'
va='te'

max_iter=(400 300 100)
eps=1e-5

lambda=(0.5 0.05 0.005)

log_path="logs/nf"
mkdir -p $log_path

# Initialization read function
matlab -nodisplay -nosplash -nodesktop -r "make;exit;"

#task(){
for i in ${!lambda[*]}; do
    lambda_U=${lambda[$i]}
    lambda_V=${lambda[$i]}

    log_name="log_nf_${tr}_${va}_gpu_${lambda_U}_${d}_${max_iter[$i]}.txt"
    matlab -nodisplay -nosplash -nodesktop -r "epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter[$i]};run('example_gpu.m');exit;"  > $log_path/$log_name 
done
#
#task
#wait

#task | xargs -0 -d '\n' -P 5 -I {} matlab {} &
