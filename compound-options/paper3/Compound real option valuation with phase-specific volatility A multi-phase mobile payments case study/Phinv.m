function z = Phinv(w)
%
%  Standard statistical inverse normal distribution
%
z = sqrt(2)*erfinv( 2*w - 1 );
return
%
% end phinv
