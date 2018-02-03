function [C, sigma] = dataset3Params_2(X, y)

Cniz = [0.01; 0.03; 0.1; 0.3; 1];
sigmaNiz = [0.01; 0.03; 0.1; 0.3; 1];
rezultat = 0;
parametriNiz = [];
oceneNiz = [];
Xdim = size(X, 1);
k = 10;
brojac = 1;
for i = 1:length(Cniz),
  for j = 1:length(sigmaNiz),
    indeksi = crossvalind('Kfold', Xdim, k);
    greska = 0;
    for l = 1:k,
      testInd = (indeksi == l);
      trainInd = ~testInd;
      model = svmTrain(X(trainInd, :), y(trainInd), Cniz(i), @(x1, x2) gaussianKernel(x1, x2, sigmaNiz(j)));
      rezultatTest = svmPredict(model, X(testInd, :));
      greska = sum(rezultatTest ~= y(testInd)) / size(testInd, 1)
      fprintf('Trenutna iteracija ---> %d\n', brojac++);
    end
      greskaProsecna = greska / k;
      fprintf('C = %d, sigma = %d\n', Cniz(i), sigmaNiz(j));
      oceneNiz = [oceneNiz; greskaProsecna];
      parametriNiz = [parametriNiz; Cniz(i), sigmaNiz(j)];
  end
end
[ocenaMin, indeksMin] = min(oceneNiz);
C = parametriNiz(indeksMin, 1);
sigma = parametriNiz(indeksMin, 2);
greskaMin = oceneNiz(indeksMin)
fprintf('\nOptimalna rešenja:\nC = %d\nsigma = %d\nGreška = %d', C, sigma, greskaMin);
% =========================================================================
end
