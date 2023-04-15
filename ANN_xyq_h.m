%load file_h.mat
% %import data
% clear all
% clc
% close all
% 
% aa=csvread('resultsup560.csv',1,0);
% a=aa;
% %a=aa((1693693:1900000),:);
% %% Normalization
% %Min & Max
% max_Q=max(a(:,2)); min_Q=min(a(:,2));
% max_x=max(a(:,5)); min_x=min(a(:,5));
% max_y=max(a(:,6)); min_y=min(a(:,6));
% max_d=max(a(:,9)); min_d=min(a(:,9));
% max_z=max(a(:,11)); min_z=min(a(:,11));
% %Normalized Data
% % 1st ANN
% %x
% ANN1_input(:,1)=(a(:,5)-min_x)/(max_x-min_x);
% %y
% ANN1_input(:,2)=(a(:,6)-min_y)/(max_y-min_y);
% %Flow (Q)
% ANN1_input(:,3)=(a(:,2)-min_Q)/(max_Q-min_Q);
% %Depth (d)
% ANN1_output(:,1)=(a(:,9)-min_d)/(max_d-min_d);
% 
% ANN1_inputT=ANN1_input';
% ANN1_outputT=ANN1_output';
% 
% %%
% rng(8)
%%
%options = trainingOptions('gdm');
    %'LearnRateSchedule','piecewise', ...
    %'LearnRateDropFactor',0.2, ...
    %'LearnRateDropPeriod',5, ...
    %'MaxEpochs',20, ...
    %'MiniBatchSize',64, ...
    %'Plots','training-progress')
tic

% net = fitnet([150 100 50 30],'traingdm');  % number of neourns %
% %net = fitnet(1you00,'traingdx');  % number of neourns %
% net.trainParam.max_fail = 100;
% net.trainParam.epochs =40000; % number of epochs
% net.divideParam.trainRatio = 70/100; % training %
% net.divideParam.valRatio = 20/100; % validation %
% net.divideParam.testRatio = 10/100; % testing %
% view(net)
% 
% X=ANN1_inputT;
% T=ANN1_outputT;

[net,tr] = train(net,X,T,'useParallel','yes','useGPU','yes','showResources','yes'); %--> how to change the percentage od train% test% validation%
%[net,tr] = train(net,X,T,'useParallel'); %--> how to change the percentage od train% test% validation%

toc
% nntraintool

