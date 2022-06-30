function [aa bb]=aa_bb_f(V,Vbar,r,vol,hh)

bb=(log(V/Vbar)+(r-vol^2/2)*hh)/(vol*(hh^0.5));


end

