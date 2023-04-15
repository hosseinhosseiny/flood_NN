%%
% Plot for h performance
figure,
tt=(1:80002);
semilogy(tt,perf_totl_h,'b',tt,vperf_totl_h,'k',tt,tperf_totl_h,'r','LineWidth',2)
xlim([1,80000])
legend ('Training','Validation', 'Test')
grid on