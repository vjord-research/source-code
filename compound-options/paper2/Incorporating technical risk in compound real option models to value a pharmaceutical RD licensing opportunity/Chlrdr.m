function [ c, ap, bp ] = Chlrdr( R, a, b )
%
%  Computes permuted lower Cholesky factor c for R which may be singular, 
%   also permuting integration limit vectors a and b.
%
ep = 1e-100; % singularity tolerance;
%
[n,n] = size(R); c = R; ap = a; bp = b; y = zeros(n,1); sqtp = sqrt(2*pi);
for k = 1 : n
   im = k; ckk = 0; dem = 1; s = 0; 
   for i = k : n 
       if c(i,i) > eps
          cii = sqrt( max( [c(i,i) 0] ) ); 
          if i > 1, s = c(i,1:k-1)*y(1:k-1); end
          ai = ( a(i)-s )/cii; bi = ( b(i)-s )/cii; de = Phi(bi) - Phi(ai);
          if de <= dem, ckk = cii; dem = de; am = ai; bm = bi; im = i; end
       end
   end
   if im > k
      tv = ap(im); ap(im) = ap(k); ap(k) = tv;
      tv = bp(im); bp(im) = bp(k); bp(k) = tv;
      c(im,im) = c(k,k); 
      t = c(im,1:k-1); c(im,1:k-1) = c(k,1:k-1); c(k,1:k-1) = t; 
      t = c(im+1:n,im); c(im+1:n,im) = c(im+1:n,k); c(im+1:n,k) = t; 
      t = c(k+1:im-1,k); c(k+1:im-1,k) = c(im,k+1:im-1)'; c(im,k+1:im-1) = t'; 
   end
   if ckk > ep*k^2
      c(k,k) = ckk; c(k,k+1:n) = 0;
      for i = k+1 : n
         c(i,k) = c(i,k)/ckk; c(i,k+1:i) = c(i,k+1:i) - c(i,k)*c(k+1:i,k)';
      end
      y(k) = ( exp( -am^2/2 ) - exp( -bm^2/2 ) )/( sqtp*dem ); 
   else
      c(k:n,k) = 0; y(k) = 0;
   end
end
return
%
% end chlrdr
