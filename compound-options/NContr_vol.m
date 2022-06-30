function [P,Vbar]=NContr_vol(V,r,vol,t00,t_i,alpha_i,beta_i,p_i,d_i,init,MaxStep,StepSize)

% the function computes the price of an n-fold compound contraction option
% Input: 
% 1)V-the current price of the underlying asset
% 2)r-the interest rate (generally should be given ready in a log form)
% 3)vol - volatility of the underlying asset
% 4)t00 - today 
% 6)alpha_i - (n,1) vector of exercise dates
% 7)beta_i - (n,1) vector of exercise dates
% 8)p_i - (n,1) vector of technical probabilities risk
% 9)init - number that adjusts initial values in finding the zeros of
% the nonlinear functions below. This number is subjest to change. A rule of thumb is
% to put init to be equal to 1. If it does not work
% and the function collapses init or qq below should be adjusted manually.
% Output: 
% P - price
% Vbars - the cut-off values from the theoretical formuala

format long
dimm=length(p_i(:,1));d_ii=1-d_i;
t=t_i;alpha=alpha_i;beta=zeros(dimm,1);

V=prod(p_i(:,1))*V*prod(d_ii(:,1));

for i=1:1:dimm %computing the modified strikes
    beta(dimm-i+1,1)=beta_i(dimm-i+1,1)*prod(p_i([1:dimm-i+1],1));
end

K=beta./(1-alpha_i);
n=length(t); %computes the dimension of the problem
qq=init;
initial=(K(n,1)*qq)^(0.5);
Vbar=zeros(n,1);
Vbar(n,1)=K(n,1); 

NMax=MaxStep; %NMax times increas, NMax times decrease by a step of step 
step=StepSize;

up=1:step:NMax;down=1:-step:0;
shift=[down(end:-1:1) up(2:end)];
lnt=length(shift);

if n>1
    for s=n:-1:2 % loop for finding Vbar_s
        current=s-1;
        t0=t(current,1);
        %y=@(x)(Price_s(s,x,Vbar,r,vol,t0,t,K))-K(s-1,1);%Price_s as function of V
        options=optimset('Display','off');    
        %options=optimset('Display','notify');
        %[x,fval,exitflag,output   
        exitflag1=10;it=1;
        while (exitflag1~=1 && it<lnt)        
            aa=ceil((it-1)/2)*((-1)^(it));
            itt=(ceil((lnt-1)/2))+aa;
            initial=initial*shift(itt);         
            [x1,fval1,exitflag1,output1]=fzero(@Price_ssc_vol,initial,options,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize);%finding Vbar_s-1         
            it=it+1;
        end
         if exitflag1==1 
            Vbar(s-1,1)=x1*x1;
            initial=((Vbar(s-1,1))*qq)^(0.5); 
            indic=1;
        else
            disp('No convergence, please try another initial value');
            indic=0;
            break
         end
    end
    if indic==1
        k=1; % finding the price
        [P]=Price_sc_vol(V,k,Vbar,r,vol,t00,t,alpha,beta,init,MaxStep,StepSize);  
    else
        P='ERROR';
    end
else
    kk=1; % finding the price
    [P]=Price_sc_vol(V,kk,Vbar,r,vol,t00,t,alpha,beta,init,MaxStep,StepSize);
end
    

