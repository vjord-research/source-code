function ww=Price_ssI_vol(V,s,Vbar,r,vol,t0,t,gamma,K,d,init,MaxStep,StepSize)
%ww=Price_sI_vol(V*V,s,Vbar,r,vol,t0,t,gamma,init,MaxStep,StepSize)-beta(s-1,1)/(alpha(s-1,1)-1);

gam=sum(gamma([1:s-1],1))+gamma(s-1,1);  

ww=Price_sI_vol(V*V,s,Vbar,r,vol,t0,t,gamma,d,init,MaxStep,StepSize)+gam*V*V-K(s-1,1)+gamma(s-1,1)*d(s-1,1)*V*V;


