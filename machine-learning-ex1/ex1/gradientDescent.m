function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
delta=zeros(length(theta));
for k=1:length(theta)
temp=0;
  for i=1:m
    temp2=0;
    for j=1:length(theta)
      temp2=temp2+theta(j)*X(i,j);
    end
    temp=temp+((temp2-y(i))*X(i,k));
  end
temp=temp/m;
temp=temp*alpha;
delta(k)=temp;
end
for k=1:length(theta)
  theta(k)=theta(k)-delta(k);
end






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
