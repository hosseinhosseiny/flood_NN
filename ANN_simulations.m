% Hossein Hosseiny
%import data
clear all
clc
close all
%%
%load net
load file_z
load file_h
data_orig=csvread('Result_Q600_mesh1m_f.csv',1,0);
% changed to ANN input formats
data(:,2)=data_orig(:,20);%WSE
data(:,5)=data_orig(:,3);%X
data(:,6)=data_orig(:,4);%Y
data(:,9)=data_orig(:,7);%h
data(:,10)=data_orig(:,8);%WSE
data(:,11)=data_orig(:,9);%z


%% Normalization
%Min & Max
%Normalized Data
% 1st ANN
%x
ANNh600_input(:,1)=(data(:,5)-min_x)/(max_x-min_x);
%y
ANNh600_input(:,2)=(data(:,6)-min_y)/(max_y-min_y);
% %Flow (Q)
ANNh600_input(:,3)=(data(:,2)-min_Q)/(max_Q-min_Q);
% Depth (z)
% ANN1_output(:,1)=(a(:,11)-min_z)/(max_z-min_z);
ANNz600_input= [ANNh600_input(:,1),ANNh600_input(:,2)];%[x,y]
ANNh600_inputT=ANNh600_input';
ANNz600_inputT=ANNz600_input';
%% simulations
z_sim_600T=sim(netz,ANNz600_inputT);
h_sim_600T= sim(net,ANNh600_inputT);
z_sim_600=z_sim_600T';
h_sim_600=h_sim_600T';
%%
z_ANN_600= (z_sim_600 *(max_z-min_z)+ min_z);
h_ANN_600=(h_sim_600 * (max_d - min_d)+min_d);
%h_ANN_600 (h_ANN_600<0)=0;
wse_ANN_600=h_ANN_600+z_ANN_600;
wse_model_600=data(:,8);
er_wse_600= wse_ANN_600-wse_model_600;% Error
er_z_600 = z_ANN_600 - data(:,9);
er_h_600= h_ANN_600 - data(:,7);
sq_er_wse_600= er_wse_600.^2;   % Squared Error wse
sq_er_z_600= er_z_600.^2;   % Squared Error z
sq_er_h_600= er_h_600.^2;   % Squared Error h

RMSE_wse = sqrt(mean(sq_er_wse_600))  % Root Mean Squared Error wse
RMSE_z = sqrt(mean(sq_er_z_600))  % Root Mean Squared Error z
RMSE_h = sqrt(mean(sq_er_h_600))  % Root Mean Squared Error h
%%
hold on
scatter3(data(:,5), data(:,6), data(:,9),1,'green') % x h real
scatter3(data(:,5), data(:,6), h_ANN_600,1, 'red')% x y h ANN
hold off

figure
hold on
scatter3(data(:,5), data(:,6), data(:,11),1,'black') % x y z Real
scatter3(data(:,5), data(:,6), z_ANN_600,1, 'red')% x y z ANN
hold off

%legend( [aa(:,5), aa(:,6), aa(:,11),aa(:,5), aa(:,6), z_ANN_600], {'Model', 'ANN'});
%pcshowpair(aa(:,5), aa(:,6), aa(:,11),aa(:,5), aa(:,6), z_ANN_600)


