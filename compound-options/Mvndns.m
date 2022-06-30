function p = Mvndns( n, ch, ci, dci, x, a, b )
%
%  Transformed integrand for computation of MVN probabilities. 
%
y = zeros(n-1,1); s = 0; c = ci; dc = dci; p = dc; 
for i = 2 : n
   y(i-1) = Phinv( c + x(i-1)*dc ); s = ch(i,1:i-1)*y(1:i-1); 
   ct = ch(i,i); ai = a(i) - s; bi = b(i) - s;
   if ai > -9*ct, if ai < 9*ct, c=Phi(ai/ct); else, c=1; end, else c=0; end
   if bi > -9*ct, if bi < 9*ct, d=Phi(bi/ct); else, d=1; end, else d=0; end
   dc = d - c; p = p*dc; 
end 
return
%
% end mvndns
