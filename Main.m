% Main file
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% Show the first 100 images
display_network(images(:,1:100));
disp(labels(1:100));