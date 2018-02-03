clear ; close all; clc

% =============== Part 1: Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

load('dataset1.mat');
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% ==================== Part 2: Training Linear SVM ====================

load('dataset1.mat');

fprintf('\nTraining Linear SVM ...\n')

C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;

% =============== Part 3: Implementing Gaussian Kernel ===============

fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

% =============== Part 4: Visualizing Dataset 2 ================

fprintf('Loading and Visualizing Data ...\n')

load('dataset2.mat');
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========

fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

load('dataset2.mat');

C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

% =============== Part 6: Visualizing Dataset 3 ================

fprintf('Loading and Visualizing Data ...\n')

load('dataset3.mat');
X = [X; Xval];
y = [y; yval];
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

load('dataset3.mat');
X = [X; Xval];
y = [y; yval];
[C, sigma] = dataset3Params(X, y, Xval, yval);

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;
