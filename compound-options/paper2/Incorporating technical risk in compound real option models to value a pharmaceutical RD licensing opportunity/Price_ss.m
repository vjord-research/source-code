function ww=Price_ss(V,s,Vbar,r,vol,t0,t,K)
ww=Price_s(V*V,s,Vbar,r,vol,t0,t,K)-K(s-1,1);