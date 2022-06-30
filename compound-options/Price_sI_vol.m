function [P_s]=Price_sI_vol(V,s,Vbar_i,r,vol_i,t0,t_i,gamma_i,d_i,init,MaxStep,StepSize)
% the function computes the price of the option under focus
rand('seed',0);
n=length(t_i);
nn=n-s+1;

h_ii=zeros(1,nn);
t_ii=zeros(1,nn);
gamma_ii=zeros(1,nn);
Vbar_ii=zeros(1,nn);
vol_ii=zeros(1,nn);
d_ii=zeros(1,nn);
C=zeros(1,nn);
g=zeros(1,nn);
K_ii=zeros(nn,nn);

for j=nn:-1:1
    h_ii(1,nn-j+1)=t_i(s+nn-j,1)-t0;%n+1-j
    t_ii(1,nn-j+1)=t_i(s+nn-j,1);
    gamma_ii(1,nn-j+1)=gamma_i(s+nn-j,1);
    vol_ii(1,nn-j+1)=vol_i(s+nn-j,1);
    d_ii(1,nn-j+1)=d_i(s+nn-j,1);
    Vbar_ii(1,nn-j+1)=Vbar_i(s+nn-j,1); 
end;

for m=0:1:nn-1
    K_ii(m+1,nn-m)=Vbar_ii(1,nn-m);
    for jj=1:1:nn-m-1 %number of terms, the last is already counted above
        diii=1-d_ii(1,[nn-m-jj+1:1:nn-m]); %only going to 2 due to the formula
        delt=prod(diii);
        K_ii(m+1,nn-m-jj)=Price_s_vol(delt*Vbar_ii(1,nn-m-jj),1,Vbar_ii(1,[nn-m-jj+1:1:nn-m])',r,vol_ii(1,[nn-m-jj+1:1:nn-m])',t_ii(1,nn-m-jj)',t_ii(1,[nn-m-jj+1:1:nn-m])',K_ii(1,[nn-m-jj+1:1:nn-m])');
    end;
    if m==0
        g(1,m+1)=gamma_ii(1,nn-m);
    else
        g(1,m+1)=sum(gamma_ii(1,[1:nn-m]))+gamma_ii(1,nn-m);
    end;    

    C(1,m+1)=g(1,m+1)*NCall_vol(V,r,vol_ii(1,[1:1:nn-m])',t0,t_ii(1,[1:1:nn-m])',K_ii(m+1,[1:1:nn-m])',ones(nn-m,1),d_ii(1,1:1:nn-m)',init,MaxStep,StepSize);         
end;
P_s=sum(C(1,:));    


            

