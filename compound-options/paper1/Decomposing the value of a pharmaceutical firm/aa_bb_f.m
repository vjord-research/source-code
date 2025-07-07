function [aa,bb]=aa_bb_f(V,Vbar,r,vol,t,t0)
hh=t-t0;
bb=(log(V/Vbar)+(r-vol^2/2)*hh)/(vol*(hh^0.5));
aa=bb+vol*(hh^0.5);

end

