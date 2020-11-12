% Test file
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% Show the first image
% display_network(images(:,1));
% disp(labels(1));

% Show the one pixel
display_network(images(:,1:8));
disp(labels(1));

% Show the first 10 images
% display_network(images(:,1:10));
% disp(labels(1:10));

% Show the first 100 images
% display_network(images(:,1:100));
% disp(labels(1:100));