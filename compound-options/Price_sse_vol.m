function ww=Price_sse_vol(V,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize)
ww=Price_se_vol(V*V,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize)-beta(s-1,1)/(alpha(s-1,1)-1);

