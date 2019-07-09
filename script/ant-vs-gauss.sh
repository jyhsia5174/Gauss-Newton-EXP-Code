d=40
#tr='ml.tr'
#va='ml.te'
#tr='ml.tr.10w'
#va='ml.tr.10w'
tr='nf.tr'
va='nf.te'

log_path="logs/$tr.$va.$d"
mkdir -p $log_path

matlab -nodisplay -nosplash -nodesktop -r "make;exit;" 
lambda_U=0.05
lambda_V=0.05
log_name="$tr.$va.$d.$lambda_U.$lambda_V"
matlab -nodisplay -nosplash -nodesktop -r "lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';run('example.m');exit;" > $log_path/$log_name& 


#lambda_U=0.5
#lambda_V=0.5
#log_name="$tr.$va.$d.$lambda_U.$lambda_V"
#matlab -nodisplay -nosplash -nodesktop -r "lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';run('example.m');exit;" > $log_path/$log_name& 


#lambda_U=0.005
#lambda_V=0.005
#log_name="$tr.$va.$d.$lambda_U.$lambda_V"
#matlab -nodisplay -nosplash -nodesktop -r "lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr='${tr}';va='${va}';run('example.m');exit;" > $log_path/$log_name& 
