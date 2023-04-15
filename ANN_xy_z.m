%import data
clear all
clc
close all

aa=csvread('resultsup560.csv',1,0);
 a=aa;
 a3=aa(1:47047,:);

%% Normalization
%Min & Max
max_x=max(a(:,5)); min_x=min(a(:,5));
max_y=max(a(:,6)); min_y=min(a(:,6));
max_z=max(a(:,11)); min_z=min(a(:,11));
%Normalized Data
% 1st ANN
%x
ANN2_input(:,1)=(a3(:,5)-min_x)/(max_x-min_x);
%y
ANN2_input(:,2)=(a3(:,6)-min_y)/(max_y-min_y);
% Depth (z)
ANN2_output(:,1)=(a3(:,11)-min_z)/(max_z-min_z);

ANN2_inputT=ANN2_input';
ANN2_outputT=ANN2_output';


%%
rng(8)
%%
% net = fitnet([100 60 30 10],'traingdm');  % number of neourns %
netz = fitnet([100 50],'traingdm');  % number of neourns %
netz.trainParam.max_fail = 100;
netz.trainParam.epochs =40000; % number of epochs
netz.divideParam.trainRatio = 70/100; % training %
netz.divideParam.valRatio = 20/100; % validation %
netz.divideParam.testRatio = 10/100; % testing %
% view(net)
Xz=ANN2_inputT;
Tz=ANN2_outputT;
[netz,trz] = train(netz,Xz,Tz,'useParallel','yes','useGPU','yes','showResources','yes'); %--> how to change the percentage od train% test% validation%
% nntraintool
%%
% figure,
% tt=(1:80002);
% semilogy(tt,perf_totl_h,'b',tt,vperf_totl_h,'k',tt,tperf_totl_h,'r','LineWidth',2)
% xlim([1,80000])
% legend ('Training','Validation', 'Test')
% grid on