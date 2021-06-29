lambda_U=0.05
lambda_V=0.05
d=40
tr="'ml.tr'"
va="'ml.te'"
matlab -nodisplay -nosplash -nodesktop -r "lambda_U=${lambda_U};lambda_V=${lambda_V};d=${d};tr=${tr};va=${va};run('example.m');exit;" 
