clc
%clear all
%load file_h2.mat
%load file_z.mat
a300=csvread('Result_Q300_mesh1m.csv',1,0);
ANNh300_input(:,1)=(a300(:,3)-min_x)/(max_x-min_x);
%y
ANNh300_input(:,2)=(a300(:,4)-min_y)/(max_y-min_y);
%Flow (Q)
ANNh300_input(:,3)=(a300(:,20)-min_Q)/(max_Q-min_Q);
%Depth (h)
y_meas_h(:,1)=(a300(:,7)-min_d)/(max_d-min_d);
y_meas_h_T=y_meas_h';

ANNh300_inputT=ANNh300_input';
% ANN1_outputT=ANN1_output';
ypred= net(ANNh300_inputT);
er_h=(y_meas_h_T-ypred)';
perf_h = mae(er_h)
%ypred_f= (ypred' * (max_d-min_d) ) + min_d;% converting back 
ypred_f= (ypred' * (4.3536) ) + min_d;% converting back 
%%
%--------Prediction for z

ANNz300_input(:,1)=(a300(:,3)-min_x)/(max_x-min_x);
%y
ANNz300_input(:,2)=(a300(:,4)-min_y)/(max_y-min_y);

y_meas_z(:,1)=(a300(:,9)-min_z)/(max_z-min_z);
%y_meas_z_T=y_meas_z';

ANNz300_inputT=ANNz300_input';
% ANN1_outputT=ANN1_output';
ypredz= netz(ANNz300_inputT);
erz=y_meas_z-ypredz';
perf_z = mae(erz)
ypredz_f= (ypredz' * (max_z-min_z) ) + min_z;% converting back 

expo300=[a300(:,3),a300(:,4),a300(:,9),a300(:,7),ypredz_f, ypred_f];
T = array2table(expo300,'VariableNames',{'x','y','z','h_iric','z_ann','h_ann'});
writetable(T,'expo300_z_h_foad.csv','Delimiter',',');
%csvwrite('expo300_z_h.csv',expo300)