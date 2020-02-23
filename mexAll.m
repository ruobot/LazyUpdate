clear;
clc;
mex MiG.cpp -largeArrayDims
mex ProxSVRG.cpp -largeArrayDims 
mex SDAMM.cpp -largeArrayDims
mex Katyusha.cpp -largeArrayDims
mex DASVRDA.cpp -largeArrayDims