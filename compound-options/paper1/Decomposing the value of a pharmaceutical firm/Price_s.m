function [P_s,hh,aa,bb,NN,w2,w1]=Price_s(V,s,Vbar,r,vol,t0,t,K)
% this function computes Ps, given s,V,Var,r,vol,t0, t vector and K vector
%V=prod(p_i(:,1))*V;
rand('seed',0);
nn=length(t);   
NN=zeros(1,nn);
aa=zeros(1,nn);aa0=-inf.*ones(1,nn);
bb=zeros(1,nn);bb0=-inf.*ones(1,nn);
hh=zeros(1,nn);
for j=nn:-1:s
    hh(1,nn-j+1)=t(s+nn-j,1)-t0;
    bb(1,nn-j+1)=(log(V/Vbar(s+nn-j,1))+(r-vol^2/2)*hh(1,nn-j+1))/(vol*(hh(1,nn-j+1)^0.5));
    aa(1,nn-j+1)=bb(1,nn-j+1)+vol*(hh(1,nn-j+1)^0.5);
    w2=F(s,nn-j+1,t,t0);
    NN(1,nn-j+1)=Qsimvn(nn-j+1,w2,bb0(1,1:nn-j+1),bb(1,1:nn-j+1));
end
w1=F(s,nn-s+1,t,t0);
u=exp(-r*hh(1,1:nn-s+1)).*(K(s:nn,1))';
NK=u*(NN(1,1:nn-s+1))';
P_s=V*Qsimvn(nn-s+1,w1,aa0(1,1:nn-s+1),aa(1,1:nn-s+1))-NK;


