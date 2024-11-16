function [SLDiff] = PortfolioLDiff(w,inputdata,x,y,ni,indic)   % the optimization objective function

    % input: w - weights, inputdata - data, x - vector of x grid, y - vector of y grid, ni - risk aversion, indic - indicator if we have a GS or GINI optimization
    % output - optimization criterion value                                                                            
    x(1,end)=0.99;
    T=length(inputdata(:,1));
    switch indic
        case 'GS1' %the same as GINI. It is controlled by the x and y input
            PortR=inputdata*w';
            L=zeros(1,T);
            for k=1:1:T
                L(1,k)=sum_smallest(PortR(:,1), k)./sum(PortR(:,1));
            end
            LDiff=ni*(ni-1)*abs(y-L).*((1-x).^(ni-2));
        case 'GS2'
            PortR=abs(inputdata*w');
            L=zeros(1,T);
            for k=1:1:T
                L(1,k)=sum_smallest(PortR(:,1), k)./sum(PortR(:,1));
            end
            LDiff=ni*(ni-1)*abs(y-L).*((1-x).^(ni-2));
        case 'GINI'
            PortR=inputdata*w';
            L=zeros(1,T);
            for k=1:1:T
                L(1,k)=sum_smallest(PortR(:,1), k)./sum(PortR(:,1));
            end
            LDiff=ni*(ni-1)*abs(y-L).*((1-x).^(ni-2));
     end
    
    SLDiff=sum(PortR(:,1))*trapz(x,LDiff)/trapz(x,y);
end