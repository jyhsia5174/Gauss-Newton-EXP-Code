d=40
tr='ratings.dat.trva'
va='ratings.dat.te'

max_iter=500

log_path="logs/$tr.$va.$d"
mkdir -p $log_path

#matlab -nodisplay -nosplash -nodesktop -r "make;exit;" 

task(){
for l in 1e-1  5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5
do
    lambda_U=$l
    lambda_V=$l
    log_name="$tr.$va.$d.$lambda_U.$lambda_V"
    echo "matlab -nodisplay -nosplash -nodesktop -r \"lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';max_iter=${max_iter};run('example.m');exit;\" > $log_path/$log_name& "
done
}

task

#task | xargs -P 4 -I {}  sh -c {} &
