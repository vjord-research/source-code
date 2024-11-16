function [w,ret,risk] = RiskM(inputData,PNum, risk, opt, misc, pos,LeverageD,LeverageU)        

% input: inputData - the data; PNum, risk, opt, misc, pos - parameters from PortfolioOpt.m        
% output: w - optimal weights, ret - optimal return, risk - optimal risk measure

optMethod=opt(1,1);optVendor=opt(1,2);optSubMethod=opt(1,3);
[T,N] = size(inputData);
mu = mean(inputData);
switch risk 
    case 'CVAR'
        switch optMethod 
            case 'XYZ' %other methods than the Matlab build-in ones can be added                
        end
    case 'GINI'
        indic="GINI"; ni=misc(1,1);
        switch optMethod 
            case 'nonlinear'
                xi=linspace(0,1,T);
                switch pos
                    case '+'
                        l_b = zeros(1,N);
                        l_u = [];
                    case '-'
                        l_b = LeverageD.*ones(1,N);
                        l_u = LeverageU.*ones(1,N);
                end
                switch optVendor
                    case 'matlab'
                        x0=(1/N).*ones(1,N); 
                        Aeq = ones(1,N);beqt=1;
                        A=-mu;b=-0.0000001;
                        GS=@(w)PortfolioLDiff(w,inputData,xi,xi,ni,indic);
                        options = optimoptions('fmincon','display','none');
                        [Sol_minGS,Risk_minGS] = fmincon(GS,x0,A,b,Aeq,beqt,l_b,l_u,[],options);
                        %[Sol_minGS,Risk_minGS] = patternsearch(GS,x0,A,b,Aeq,beqt,lb,[],[],options);
                        x_0 = Sol_minGS(1:N)';
                        mu_min = mu*x_0;
                        mu_max = max(mu);
                        n = PNum;
                        mu_bar = linspace(mu_min,mu_max,n);
                        Aeq = [mu; Aeq];
                        x = NaN(N,length(mu_bar));
                        RiskGS = NaN(length(mu_bar),1);
                        RiskGS(1,1)=Risk_minGS/T;
                        x(:,1) = x_0;
                        for i=2:n
                            beq=[mu_bar(i);beqt];
                            options = optimoptions('fmincon','display','none');%x(:,i-1)'
                            [SolGS,riskGSi] = fmincon(GS,x(:,i-1)',[],[],Aeq,beq,l_b,l_u,[],options);
                            RiskGS(i,1)=riskGSi/T;
                            x(:,i) = SolGS(1:N)';
                        end
                        w=x;
                        risk=RiskGS(:,1);
                        ret=mu_bar;
                end 
        end
    case 'Golden Section'
        upTailPareto=misc(1,1);upTailPower=misc(1,2);
        downTailPareto=misc(2,1);downTailPower=misc(2,2);
        upAlpha=misc(3,1);downAlpha=misc(3,2);
        ni=misc(4,1);
        switch misc(5,1)
            case 1
               indic="GS1";
            case 2
               indic="GS2";
        end
        [T,N] = size(inputData);
        mu = mean(inputData);
        phi1=(sqrt(5)+1)/2;phi2=(sqrt(5)-1)/2;
        PowerL = @(x) x.^phi1;ParetoL = @(x) 1-(1-x).^phi2;
        xi = linspace(0,1,T);
        x1=xi(xi<=downAlpha);x2=xi(xi>=downAlpha);x2=x2(x2<=upAlpha);x3=xi(xi>=upAlpha);
        LZdown = @(x) downTailPower*PowerL(x) + downTailPareto*ParetoL(x);
        LZup = @(x) upTailPower*PowerL(x) + upTailPareto*ParetoL(x);
        y1=LZdown(x1);y3=LZup(x3);
        LZcenter = @(x) ((y3(1,1)-y1(1,end))/(x3(1,1)-x1(1,end)))*(x-x3(1,1))+y3(1,1);
        y2=LZcenter(x2);
        if x1==0
            yi=[y2 y3];
            xi=[x2 x3];
        end
        if x3==0
            yi=[y1 y2];
            xi=[x1 x2];
        end
        x1n=x1~=0;x3n=x3~=0;
        x1n1=find(x1n);x3n1=find(x3n);
        lx1n1=length(x1n1);lx3n1=length(x3n1);
        if lx1n1>=3 && lx3n1>=3
            yi=[y1 y2 y3];
            xi=[x1 x2 x3];
        end
        switch optMethod           
            case 'nonlinear'        
                switch pos
                    case '+'
                        l_b = zeros(1,N);
                        l_u=[];
                    case '-'                
                        l_b = LeverageD.*ones(1,N); 
                        l_u = LeverageU.*ones(1,N); 
                end 
                switch optVendor
                    case 'matlab'
                        x0=(1/N).*ones(1,N); 
                        Aeq = ones(1,N);beqt=1;
                        A=-mu;b=-0.0000001;
                        GS=@(w)PortfolioLDiff(w,inputData,xi,yi,ni,indic);
                        options = optimoptions('fmincon','display','none');
                        [Sol_minGS,Risk_minGS] = fmincon(GS,x0,A,b,Aeq,beqt,l_b,l_u,[],options);
                        x_0 = Sol_minGS(1:N)';
                        mu_min = mu*x_0;
                        mu_max = max(mu);
                        n = PNum;
                        mu_bar = linspace(mu_min,mu_max,n);
                        Aeq = [mu; Aeq];
                        x = NaN(N,length(mu_bar));
                        RiskGS = NaN(length(mu_bar),1);
                        RiskGS(1,1)=Risk_minGS/T;
                        x(:,1) = x_0;
                        for i=2:n
                            beq=[mu_bar(i);beqt];
                            options = optimoptions('fmincon','display','none');
                            [SolGS,riskGSi] = fmincon(GS,x(:,i-1)',[],[],Aeq,beq,l_b,l_u,[],options);
                            RiskGS(i,1)=riskGSi/T;
                            x(:,i) = SolGS(1:N)';
                        end
                        w=x;
                        risk=RiskGS(:,1);
                        ret=mu_bar;
                end 
        end
end


