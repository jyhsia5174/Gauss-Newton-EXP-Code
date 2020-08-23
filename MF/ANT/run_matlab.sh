#!/bin/bash

matlab="/usr/local/bin/matlab"

${matlab} -nodisplay -nodesktop -nosplash -r "diary log_nf_tr_te_gpu_0.005_20_100.txt; diary on; test_initModel_gpu; diary off, exit;"

