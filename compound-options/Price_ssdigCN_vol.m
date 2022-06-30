function ww=Price_ssdigCN_vol(V,s,Vbar,r,vol,t0,t,K)
ww=Price_sdigCN_vol(V*V,s,Vbar,r,vol,t0,t,K)-K(s-1,1);
