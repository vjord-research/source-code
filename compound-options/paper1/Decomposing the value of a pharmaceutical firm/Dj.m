function Dj=Dj(s,l,t,current,j)% computes the covariance matrix for the Greeks given 
                             % s, l, the time vector t, the current
                             % moment, and the index j from the definition of the matrix                               
F=zeros(l+1,l+1);D=zeros(l,l);
for a=1:1:l+1
    for b=(a+1):1:l+1
        F(a,b)=((t(a+s-1,1)-current)/(t(b+s-1,1)-current))^0.5;  
    end
end  
for a=1:1:l
    for b=(a+1):1:l
        if (a<=j)    
            num=F(a,b)-F(a,j+1)*F(b,j+1);
            denom=((1-F(a,j+1)*F(a,j+1))*(1-F(b,j+1)*F(b,j+1)))^(0.5);
            D(a,b)=num/denom;
        end
        if (a>j)
            if (b<=j)            
                D(a,b)=0;
            else
                num=F(a+1,b+1)-F(a+1,j+1)*F(b+1,j+1);
                denom=((1-F(a+1,j+1)*F(a+1,j+1))*(1-F(b+1,j+1)*F(b+1,j+1)))^(0.5);
                D(a,b)=num/denom;
            end
        end   
    end
end  
Dj=D+D'+eye(l);
