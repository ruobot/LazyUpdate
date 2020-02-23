%% Matlab interface

%% a9a lambda!=0
clear; clc;
dataset = 'a9a';

outer_loops = 200;
lambda = 1e-8;
alpha_mig = 0.05;
alpha_katyusha = 0.1;
alpha_sdamm = 0.028;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% a9a lambda=0
clear; clc;
dataset = 'a9a';

outer_loops = 200;
lambda = 0;
alpha_mig = 1;
alpha_katyusha = 1;
alpha_sdamm = 0.028;
minvalue = 1e-8;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% w8a lambda=1e-8
clear; clc;
dataset = 'w8a';

outer_loops = 300;
lambda = 1e-8;
alpha_mig = 0.125;
alpha_katyusha = 0.3;
alpha_sdamm = 0.04;
minvalue = 1e-12;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% phishing lambda = 1e-8
clear; clc;
dataset = 'phishing';

outer_loops = 50;
lambda = 1e-8;
alpha_mig = 0.0005;
alpha_katyusha = 0.001;
alpha_sdamm = 0.0005;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% phishing lambda = 0  ????
clear; clc;
dataset = 'phishing';

outer_loops = 50;
lambda = 0;
alpha_mig = 0.025;
alpha_katyusha = 0.025;
alpha_sdamm = 0.00875;
minvalue = 1e-10;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% breast-cancer lambda = 1e-8
clear; clc;
dataset = 'breast-cancer';

outer_loops = 150;
lambda = 1e-8;
alpha_mig = 0.0075;
alpha_katyusha = 0.02;
alpha_sdamm = 0.007;
minvalue = 1e-15;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% breast-cancer lambda = 0
clear; clc;
dataset = 'breast-cancer';

outer_loops = 150;
lambda = 0;
alpha_mig = 0.006;
alpha_katyusha = 0.02;
alpha_sdamm = 0.0068;
minvalue = 1e-15;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% cod-rna lambda = 1e-8
clear; clc;
dataset = 'cod-rna';

outer_loops = 100;
lambda = 1e-8;
alpha_mig = 0.0075;
alpha_katyusha = 0.01;
alpha_sdamm = 0.005;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% cod-rna lambda = 1e-8
clear; clc;
dataset = 'cod-rna';

outer_loops = 100;
lambda = 0;
alpha_mig = 0.0001;
alpha_katyusha = 0.0001;
alpha_sdamm = 0.00008;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% ijcnn lambda = 1e-8
clear; clc;
dataset = 'ijcnn';

outer_loops = 25;
lambda = 1e-8;
alpha_mig = 0.001;
alpha_katyusha = 0.001;
alpha_sdamm = 0.001;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% ijcnn lambda = 0
clear; clc;
dataset = 'ijcnn';

outer_loops = 25;
lambda = 0;
alpha_mig = 0.001;
alpha_katyusha = 0.001;
alpha_sdamm = 0.001;
minvalue = 1e-14;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% mushrooms lambda = 1e-8
clear; clc;
dataset = 'mushrooms';

outer_loops = 100;
lambda = 1e-8;
alpha_mig = 0.025;
alpha_katyusha = 0.025;
alpha_sdamm = 0.000875;
minvalue = 1e-12;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% mushrooms lambda = 0
clear; clc;
dataset = 'mushrooms';

outer_loops = 10;
lambda = 0;
alpha_mig = 0.03;
alpha_katyusha = 0.03;
alpha_sdamm = 0.0025;
minvalue = 1e-20;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% diabetes lambda= 1e-8
clear; clc;
dataset = 'diabetes';

outer_loops = 50;
lambda = 1e-8;
alpha_mig = 0.01;
alpha_katyusha = 0.01;
alpha_sdamm = 0.00675;
minvalue = 1e-12;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);

%% diabetes lambda= 0 
clear; clc;
dataset = 'diabetes';

outer_loops = 50;
lambda = 0;
alpha_mig = 0.01;
alpha_katyusha = 0.01;
alpha_sdamm = 0.00675;
minvalue = 1e-12;
isSave = true;
compare_data(dataset,outer_loops, lambda, alpha_katyusha, ...
    alpha_mig, alpha_sdamm, minvalue, isSave);
