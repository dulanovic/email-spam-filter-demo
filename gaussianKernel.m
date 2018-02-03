function sim = gaussianKernel(x1, x2, sigma)

x1 = x1(:); x2 = x2(:);
sim = 0;

n = length(x1);
suma = 0;
for i = 1:n,
  suma += (x1(i) - x2(i))^2;
end
sim = exp(-suma/(2*sigma^2));
% =============================================================
end
