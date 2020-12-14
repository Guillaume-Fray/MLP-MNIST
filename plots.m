% File to plot the results
clear all;
close all;

% Learning Rates tested
% Corresponding Error Rates obtained
% learn_rate = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.1 0.15, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1, 1.5, 2, 3, 5, 10, 50];
% error_rate_lr = [86.68, 92.58, 64.82, 31.22, 27.02, 13.23, 11.01, 17.215, 19.74, 17.77, 12.3 7.29, 7.32, 9.0, 7.47, 6.4, 6.7, 6.39, 6.7, 6.85, 6.5, 6.79, 7.2, 7.1, 7.10, 10.68, 10.87, 28.36, 60.69, 91.08, 89.72];

% num_neurons = [1,2,3,5,10,15,20,30,50,60,75,85,100,150,300]; %10 Epochs, L.R = 0.3
% error_rate_neu = [90.8, 77.43, 55.65, 18.39,10.69,8.73,8.34,7.94,7.15,7.84,6.55,6.28,7.51,15.22,26.8];

num_epochs = [1,2,3,5,10,15,20,50];
error_rate_ep50 = [13.25,10.5,9.21,7.51,7.15,6.55,6.83,6.71];
% error_rate_ep20 = [13.28,11.11,10.47,9.72,8.53,7.96,8.59,8.48];
% error_rate_ep15 = [13.83,12.96,11.73,10.73,9.65,9.97,9.7,9.22];
% error_rate_ep10 = [18.88,14.82,14.35,12.48,10.45,10.57,10.71,9.59];



% disp(size(learn_rate));
% disp(size(error_rate_lr));


% Pick your variables HERE        <------ 1
X = num_epochs;
Y = error_rate_ep50;

scatter((X),(Y),'filled');
hold on;

% Pick your axes HERE             <------ 2
xlabel('Nb of Epochs') ;
ylabel('Error Rate ( % )') ;

% Choose your Xmin & Xmax HERE    <------ 3
xMin = 0;
xMax = 50;
xlim([xMin xMax]);
yMin = 0;
yMax = 20;
ylim([yMin yMax]);

for i = 1:(length(X)-1)
    x1 = (X(i));
    y1 = (Y(i));
    x2 = (X(i+1));
    y2 = (Y(i+1));
    plot([x1,x2], [y1, y2], 'r-', 'LineWidth', 2);
    set(gca,'FontSize',20)
end


