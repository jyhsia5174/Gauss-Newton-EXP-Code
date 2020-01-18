% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
try
    if (exist('OCTAVE_VERSION', 'builtin'))
        mex mf_read.c
    else
    % Add -largeArrayDims on 64-bit machines of MATLAB
        mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims mf_read.c
    end
catch
    fprintf('If make.m fails, please check README about detailed instructions.\n');
end
