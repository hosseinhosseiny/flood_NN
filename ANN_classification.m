%import data
clear all
clc
close all

aa=csvread('results.csv',1,0);
a=aa;

%% Normalization
%Min & Max
max_Q=max(a(:,2)); min_Q=min(a(:,2));
max_x=max(a(:,5)); min_x=min(a(:,5));
max_y=max(a(:,6)); min_y=min(a(:,6));
%max_d=max(a(:,9)); min_d=min(a(:,9));
%max_z=max(a(:,11)); min_z=min(a(:,11));
%%Dry Wet classification
dw=aa(:,7);% 0 dry and 1 is wet
%%
% %Normalized Data
% % 1st ANN
% %x
ANN1_input(:,1)=(a(:,5)-min_x)/(max_x-min_x);
%y
ANN1_input(:,2)=(a(:,6)-min_y)/(max_y-min_y);
%Flow (Q)
ANN1_input(:,3)=(a(:,2)-min_Q)/(max_Q-min_Q);
%%
rng(1); % For reproducibility
Mdl = TreeBagger(100,ANN1_input,dw,'OOBPrediction', 'on', 'Method','classification');
figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% Testing for Q=600 cms
% clear all
% clc
% close all
% load ('File_bagged_Classif')

test_data=csvread('Result_Q600_mesh1m_f.csv',1,0);

test600_input(:,1)=(test_data(:,3)-min_x)/(max_x-min_x);
test600_input(:,2)=(test_data(:,4)-min_y)/(max_y-min_y);
test600_input(:,3)=(test_data(:,20)-min_Q)/(max_Q-min_Q);
test600_res= (predict(Mdl,test600_input));
test600_res_arr= ((cell2mat(test600_res)));
wetdry600=str2num(test600_res_arr(:,1));
error600= abs(test_data(:,5)- wetdry600);
sum(error600)
[numRows,numCols] = size(test_data);
error_Percentage= sum(error600)/(numRows) *100