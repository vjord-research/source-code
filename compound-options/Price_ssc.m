function ww=Price_ssc(V,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize)
ww=Price_sc(V*V,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize)-beta(s-1,1)/(1-alpha(s-1,1));
