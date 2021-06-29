% compile the libsvmread.cpp
make;

% prepare training and test data sets
[y, U, V] = libsvmread('test.tr');
