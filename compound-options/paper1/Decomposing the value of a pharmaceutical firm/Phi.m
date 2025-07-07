function p = Phi(z)
%
%  Standard statistical normal distribution
%
p = ( 1 + erf( z/sqrt(2) ) )/2;
return
%
% end phi
