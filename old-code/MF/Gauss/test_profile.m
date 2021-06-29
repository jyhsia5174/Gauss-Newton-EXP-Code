function test_profile()
data_name = 'Movielens-1m'

profile on;
test_initModel_gpu;
profile off;

profile_name=['./profile_log/' data_name];
system(['rm -rf ' profile_name]);

profsave(profile('info'),profile_name);
