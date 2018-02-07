function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
combined = (-y)'*log(h)-(1-y)'*log(1-h);
J = 1/m * sum((-y) .* log(h) - (1-y) .* log(1- h) ) + lambda/(2*m) * sum(theta(2:size(theta)).^2);
k = length(theta)-1;
n = length(theta);

X_T = X';
grad(1)=1/m.*(sum(X_T(1,:)*h-X_T(1,:)*y));

for j = 2:n
    grad(j)=(1/m).*(sum(X_T(j,:)*h-X_T(j,:)*y)+lambda.*theta(j,1));
end



% =============================================================

end
