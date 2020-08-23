
echo "check logs permission"

d=40
tr='tr'
va='te'

max_iter=100
eps=1e-5

log_path="logs/ml1m"
mkdir -p $log_path

# Initialization read function
matlab -nodisplay -nosplash -nodesktop -r "make;exit;"

#task(){
for l in 0.5 0.05 0.005
do
    lambda_U=$l
    lambda_V=$l
    log_name="log_ml10m_${tr}_${va}_gpu_${lambda_U}_${d}_${max_iter}.txt"
    matlab -nodisplay -nosplash -nodesktop -r "epsilon=${eps};lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter};run('example_gpu.m');exit;"  > $log_path/$log_name 
done
#
#task
#wait

#task | xargs -0 -d '\n' -P 5 -I {} matlab {} &
