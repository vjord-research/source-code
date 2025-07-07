function ww=Price_ss_vol(V,s,Vbar,r,vol,t0,t,K)
ww=Price_s_vol(V*V,s,Vbar,r,vol,t0,t,K)-K(s-1,1);