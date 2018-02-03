C = 10;
sigma = 3;
rezultat = 0;
load('ex6data3.mat');
Xvaldim = size(Xval, 1);
k = 10;

indeksi = crossvalind('Kfold', Xvaldim, k);
greska = 0;
for l = 1:k,
  testInd = (indeksi == l);
  trainInd = ~testInd;
  model = svmTrain(Xval(trainInd, :), yval(trainInd), C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  rezultatTest = svmPredict(model, Xval(testInd, :));
  trenutnaGreska = sum(rezultatTest ~= yval(testInd)) / size(testInd,1)
  greska += trenutnaGreska;
end
greskaProsecna = greska / k