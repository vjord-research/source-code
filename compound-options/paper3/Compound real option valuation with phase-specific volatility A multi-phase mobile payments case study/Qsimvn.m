function [ p, e ] = Qsimvn( m, r, a, b )
%
%  [ P E ] = QSIMVN( M, R, A, B )
%    uses a randomized quasi-random rule with m points to estimate an
%    MVN probability for positive semi-definite covariance matrix r,
%    with lower integration limits a and upper integration limits b. 
%   Probability p is output with error estimate e.

%
%   This function uses an algorithm given in the paper
%      "Numerical Computation of Multivariate Normal Probabilities", in
%      J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
%          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
%          Email : AlanGenz@wsu.edu
%  The primary references for the numerical integration are 
%   "On a Number-Theoretical Integration Method"
%   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
%   "Randomization of Number Theoretic Methods for Multiple Integration"
%    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
%
%   Alan Genz is the author of this function and following Matlab functions.
%
% Initialization
%
[n, n] = size(r); [ ch as bs ] = Chlrdr( r, a, b );
ct = ch(1,1); ai = as(1); bi = bs(1);
if ai > -9*ct, if ai < 9*ct, c=Phi(ai/ct); else, c=1; end, else c=0; end
if bi > -9*ct, if bi < 9*ct, d=Phi(bi/ct); else, d=1; end, else d=0; end
ci = c; dci = d - ci; p = 0; e = 0;
ns = 8; nv = max( [ m/( 2*ns ) 1 ] ); 
q = 2.^( [1:n-1]'/n) ; % Niederreiter point set generators
%
% Randomization loop for ns samples
%
for i = 1 : ns
   vi = 0; xr = rand( n-1, 1 ); 
   %
   % Loop for 2*nv quasirandom points
   %
   for  j = 1 : nv
      x = abs( 2*mod( j*q + xr, 1 ) - 1 ); % periodizing transformation
      vp =   Mvndns( n, ch, ci, dci,   x, as, bs ); 
      vp = ( Mvndns( n, ch, ci, dci, 1-x, as, bs ) + vp )/2; 
      vi = vi + ( vp - vi )/j; 
   end   
   %
   d = ( vi - p )/i; p = p + d; e = ( i - 2 )*e/i + d^2; 
end
%
e = 3*sqrt(e); % error estimate is 3 x standard error with ns samples.
return
%
% end qsimvn
