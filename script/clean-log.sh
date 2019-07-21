file=$1

cp ${file} ${file}.clean
vim -S '/tmp2/Gauss-Newton-EXP-Code/script/clean.vimscript' ${file}.clean
