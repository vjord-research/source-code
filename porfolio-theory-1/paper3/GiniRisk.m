function [w,ret,risk] = GiniRisk(inputData)
[T,n] = size(inputData);
mu = mean(inputData);
A_1 = []; 
for k=1:T
    A_1 = [A_1;inputData-repmat(inputData(k,:),T,1)];
end

A_2 = -speye(T^2); 
% T bigger than 100 -> memory problems
A = [A_1 A_2];

% sparse matrices
A_1_t= sparse(A_1);
A_2_t = sparse(A_2);
A_t = [A_1_t A_2_t];
A_t = sparse(A);
f = [zeros(1,n) ones(1,T^2)];
Aeq = [ones(1,n) zeros(1,T^2)];
beq=1;
b = zeros(T^2,1);
l_b = zeros(1,n+T^2);

b_tilde = sparse(b);
Aeq_tilde = sparse(Aeq);
l_b_tilde = sparse(l_b);

[Sol_min,~] = linprog(f,A_t,b_tilde,Aeq_tilde,beq,l_b_tilde,[]);
x_3 = Sol_min(1:n);    
eta_min = mu*x_3;
eta_max = max(mu);
N = 10;                     
eta = linspace(eta_min,eta_max,N);
Aeq = [mu zeros(1,T^2); Aeq];
beq = [0;1];
x = NaN(n,length(eta)); 
RiskGMD = NaN(length(eta),1);
x(:,1) = x_3;
for i=2:N  
    beq(1) = eta(i);
    [Sol,RiskGMD(i,1)] = linprog(f,A_t,b_tilde,Aeq,beq,l_b_tilde,[]);
    x(:,i) = Sol(1:n);
end
w=x;
risk=RiskGMD(:,1);
ret=eta;
end