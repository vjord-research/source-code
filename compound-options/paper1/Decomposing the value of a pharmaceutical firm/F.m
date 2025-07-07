function sigma=F(s,l,t,current)% computes the covariance matrix given 
                               % s, l, the time vector t, and the current
                               % moment                               
F=zeros(l,l);
for i=1:1:l
    for j=(i+1):1:l
        F(i,j)=((t(i+s-1,1)-current)/(t(j+s-1,1)-current))^0.5;
    end
end  
sigma=F+F'+eye(l);