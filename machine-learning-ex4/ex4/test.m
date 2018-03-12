z2 = Theta1 * [ones(1,m); X'];
A2 = sigmoid(z2);
z3 = Theta2 * [ones(1,m); A2];
A3 = sigmoid(z3);
h = A3;

m = size(X, 1);
num_labels = 10;
yb = zeros(m, num_labels);
for k = 1:num_labels,
  yb(:,k) = (y==k);
endfor
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
t = 9;
xt = X(t,:)';
yt = yb(t,:)';
at2 = A2(:,t);
at3 = A3(:,t);
delta3 = at3 - yt;
delta2 = Theta2' * delta3 .* [0; sigmoidGradient(z2(:,t))];
Delta2 = Delta2 + delta3 * [1; at2]';
Delta1 = Delta1 + delta2(2:end) * [1; xt]';
