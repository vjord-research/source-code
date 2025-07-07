function [P,Vbar,delta,gamma,vega,theta]=NCall_temp(V,r,vol,t00,t_i,K_i,p_i,d_i,init,MaxStep,StepSize,GreeksIndic)
% the function computes the price of an n-fold compound option
% the input is: 
% 1)V-the current price of the underlying asset
% 2)r-the interest rate (generally should be given ready in a log form)
% 3)vol - volatility of the underlying asset
% 4)t00 - today date
% 5)t - (n,1) vector of exercise dates
% 6)KK - (n,1) vector of strikes
% 7)tp - (n,1) vector of technical probabilities risk
% 8)init - number that adjusts initial values in finding the zeros of
% the nonlinear functions below. This number is subjest to change. A rule of thumb is
% to put init to be equal to 1

format long
dimm=length(p_i(:,1));d_ii=1-d_i;
t=t_i;K=zeros(dimm,1);V=prod(p_i(:,1))*V*prod(d_ii(:,1));
for i=1:1:dimm %computing the modified strikes
    K(dimm-i+1,1)=K_i(dimm-i+1,1)*prod(p_i([1:dimm-i+1],1));
end
n=length(t); %computes the dimension of the problem
qq=init;
C=zeros(n,1);
initial=(K(n,1)*qq)^(0.5);%initial=V^(0.5);%initial=(K(n-1,1)*qq)^(0.5);

Vbar=zeros(n,1);
N=zeros(n,n);N1=zeros(n,n);
a=zeros(n,n);a0=-inf.*ones(n,n);
b=zeros(n,n);b0=-inf.*ones(n,n);
h=zeros(n,n);h1=zeros(n,n); 
Vbar(n,1)=K(n,1); %Vbar_n

NMax=MaxStep; %NMax times increas, NMax times decrease by a step of step 
step=StepSize;
up=1:step:NMax;down=1:-step:0;
shift=[down(end:-1:1) up(2:end)];
lnt=length(shift);

if n>1
    for s=n:-1:2 % loop for finding Vbar_s
        current=s-1;
        t0=t(current,1);
%       y=@(x)(Price_s(s,x,Vbar,r,vol,t0,t,K))-K(s-1,1);%Price_s as function of V
%       options=optimset('Display','notify');
        options=optimset('Display','off');    
        exitflag1=10;it=1;
        while (exitflag1~=1 && it<lnt)         
            aa=ceil((it-1)/2)*((-1)^(it));
             itt=(ceil((lnt-1)/2))+aa;
            initial=initial*shift(itt);
%           exitflag1
            [x1,fval1,exitflag1,output1]=fzero(@Price_ss,initial,options,s,Vbar,r,vol,t0,t,K);%finding Vbar_s-1        
%           exitflag1
            it=it+1;     
        end
        if exitflag1==1 
            Vbar(s-1,1)=x1*x1;
            initial=((Vbar(s-1,1))*qq)^(0.5); 
            ind=1;
        else
            disp('No convergence, please try another initial value');
            ind=0;
            break
        end
        [c1,h1,a1,b1,N1,w2,w1]=Price_s(V,s,Vbar,r,vol,t0,t,K);
        C(s,1)=c1;h(s,:)=h1;a(s,:)=a1;b(s,:)=b1;N(s,:)=N1;
    end
    if ind==1
        k=1; % finding the price
        [c2,h2,a2,b2,N2,w4,w3,delta33,gamma33,vega33,theta33]=Price_sGr(V,k,Vbar,r,vol,t00,t,K,GreeksIndic);
        C(k,1)=c2;h(k,:)=h2;a(k,:)=a2;b(k,:)=b2;N(k,:)=N2;
        P=C(k,1);
    else
        P='ERROR';
    end
else
    kk=1; % finding the price
    [c2,h2,a2,b2,N2,w4,w3,delta33,gamma33,vega33,theta33]=Price_sGr(V,kk,Vbar,r,vol,t00,t,K,GreeksIndic);
    C(kk,1)=c2;h(kk,:)=h2;a(kk,:)=a2;b(kk,:)=b2;N(kk,:)=N2;
    P=C(kk,1);   
end;

if strcmp(GreeksIndic,'yes')==1
    delta=delta33;
    gamma=gamma33;
    vega=vega33;
    theta=theta33;
    
else
    if strcmp(GreeksIndic,'no')==1
        delta='na';
        gamma='na';
        vega='na';
        theta='na';
    else
        disp('Not valid greeks input indicator');       
    end
end;


    