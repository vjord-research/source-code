function ww=Price_ssdigAN_vol(V,s,Vbar,r,vol,t0,t,K)
ww=Price_sdigAN_vol(V*V,s,Vbar,r,vol,t0,t,K)-K(s-1,1);
