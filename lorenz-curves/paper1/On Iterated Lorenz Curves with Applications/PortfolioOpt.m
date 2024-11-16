clear; close all; clc; rng(0);

% Initial parameters
datamethod = "historical";              % scenarios generation method: 'simulation', 'historical', or 'loaded_data'
frec = "daily";                         % data frequency: 'daily', 'weekly', 'monthly, or 'loaded_frec' (here we may use previously saved data for better run-time)
nScenarios = 500;                       % number of scenarios
simulationMethod = "Empirical";         % simulation method: 'Normal' - based on normal distribution, 'Empirical' - based on empirical distribution using t-copula
PNum=10;                                % number of portfolios on the efficient frontier
stocksID=[1 4 7 10 13 16 19 22 25 28 31 34 37 40];       % the stocks ID for our sample portfolio as of the xls file
optCVAR=["standard" "" ""];             % 1. CVAR optimization methods: 'standard' - based on the built-in Matlab's implementation (OUR PAPER USES THIS), 2. Optimization vendors, 3. Submethods
miscCVAR=0.95;                          % 1. CVAR's miscellaneous other parameters 
optGINI=["nonlinear" "matlab" ""];
miscGINI=4;
optGS=["nonlinear" "matlab" ""];
miscGS1=[0.8 0.2;0.3 0.7;0.6 0.4; 2.5 0];    % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS1. Must be set even if not used due to a plot to be generated.
miscGS2=[1 0;0.5 0.5;0.75 0;4 0];            % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS2. Must be set even if not used due to a plot to be generated.
GS1_2 = "GS2";                          % GS method: "GS1" or "GS2" 
YorN="+";                               % "-" - negative portfolio weights allowed, "+" - only non-negative portfolio weights allowed                                               
posMV=YorN;posCVAR=YorN;posMAD=YorN;posGINI=YorN;posGS=YorN;
LBound=-0.3;UBound=1.3;                 % in case of allowed negative weights the weights bounds (i.e. the leverage)

% Data upload
Ttemp = readtable('DJI.xlsx','Sheet','data_final');
%Ttemp = readtable('STOXX50E.xlsx','Sheet','data_final');
Ttemp.Time = datetime(Ttemp.Time);
Ttemp1 = table2timetable(Ttemp);

switch frec
    case 'daily'
        T=Ttemp1;
        symbol_temp =T.Properties.VariableNames;
        symbol=symbol_temp(stocksID);
    case 'weekly'
        T = convert2weekly(Ttemp1);
        symbol_temp =T.Properties.VariableNames;
        symbol=symbol_temp(stocksID);
    case 'monthly'
        T = convert2monthly(Ttemp1);
        symbol_temp =T.Properties.VariableNames;
        symbol=symbol_temp(stocksID);
    case 'loaded_frec'
        load("DJIdata_monthly.mat","T");
        symbol_temp =T.Properties.VariableNames;
        symbol=symbol_temp(stocksID);
end

nAsset = numel(symbol);ret = tick2ret(T{:,symbol});
LB=LBound*ones(1,nAsset);UB=UBound*ones(1,nAsset);
plotAssetHist(symbol,ret);

switch datamethod    
    case 'simulation'
        switch simulationMethod
            case 'Normal'
                AssetScenarios = mvnrnd(mean(ret),cov(ret),nScenarios);
                plotAssetHist(symbol,AssetScenarios)
            case 'Empirical' 
                AssetScenarios = simEmpirical(ret,nScenarios);
                plotAssetHist(symbol,AssetScenarios)
        end              
    case 'historical'
        lret=length(ret(:,1));
        AssetScenarios=ret(lret-nScenarios+1:1:end,:);
    case 'loaded_data'
        load("DJIscenarios.mat");
end

% MV Portfolio Optimization
tic
p1 = Portfolio;
p1 = setAssetList(p1, symbol);
if posMV=="-"
    p1 = setBounds(p1, LB, UB);
    p1 = setBudget(p1, 1, 1);
else
    p1 = setDefaultConstraints(p1);
end
p1 = estimateAssetMoments(p1, ret);
w1 = estimateFrontier(p1,PNum);
toc
figure;
plotFrontier(p1,w1);
plotWeight(w1, symbol, posMV, LBound, UBound, 'Mean-Variance Portfolio ')

% CVaR Portfolio Optimization
switch optCVAR(1,1)
    case 'standard'
        tic
        p2 = PortfolioCVaR;
        p2 = setScenarios(p2, AssetScenarios);
        if posCVAR=="-"
            p2 = setBounds(p2, LB, UB);
            p2 = setBudget(p2, 1, 1);
        else
            p2 = setDefaultConstraints(p2);
        end
        CVaRlevel=miscCVAR(1,1);
        p2 = setProbabilityLevel(p2, CVaRlevel);
        w2 = estimateFrontier(p2,PNum);
        toc
        figure
        plotFrontier(p2,w2);
        plotWeight(w2, symbol, posCVAR, LBound, UBound, 'CVaR Portfolio')
end


% MAD Portfolio Optimization
tic
p3 = PortfolioMAD('Scenarios', AssetScenarios);
if posMAD=="-"
    p3 = setBounds(p3, LB, UB);
    p3 = setBudget(p3, 1, 1);
else
    p3 = setDefaultConstraints(p3);
end
w3 = estimateFrontier(p3,PNum);
toc
figure;
plotFrontier(p3,w3);
plotWeight(w3, symbol, posMAD, LBound, UBound, 'MAD Portfolio');

% Mean-Gini Portfolio Optimization
tic
[w4,retg,riskg] = RiskM(AssetScenarios,PNum, "GINI", optGINI, miscGINI,posGINI,LB,UB);
toc
plotWeight(w4, symbol, posGINI, LBound, UBound, 'Mean-Gini Portfolio');
figure;
plot(riskg,retg,'LineWidth',2)
ylabel('Mean of Portfolio Returns')
xlabel('Mean-Gini')
title('Efficient Frontier');
grid on

% Golden Section Portfolio Optimization
tic
switch GS1_2
    case "GS1"
        [w5,retgs,riskgs] = RiskM(AssetScenarios,PNum, "Golden Section", optGS, [miscGS1;1 1],posGS,LB,UB);
    case "GS2"
        [w5,retgs,riskgs] = RiskM(AssetScenarios,PNum, "Golden Section", optGS, [miscGS2;2 2],posGS,LB,UB);
end
toc
plotWeight(w5, symbol, posGS, LBound, UBound, 'Golden Section Portfolio');
figure;
plot(riskgs,retgs,'LineWidth',2)
ylabel('Mean of Portfolio Returns')
xlabel('Golden Section')
title('Efficient Frontier');
grid on

benchmark=p3;
pRet1 = estimatePortReturn(benchmark,w1);
pRisk1 = estimatePortRisk(benchmark,w1);
pRet2 = estimatePortReturn(benchmark,w2);
pRisk2 = estimatePortRisk(benchmark,w2);
pRet3 = estimatePortReturn(benchmark,w3);
pRisk3 = estimatePortRisk(benchmark,w3);
pRet4 = estimatePortReturn(benchmark,w4);
pRisk4 = estimatePortRisk(benchmark,w4);
pRet5 = estimatePortReturn(benchmark,w5);
pRisk5 = estimatePortRisk(benchmark,w5);

plotWeight2(w1, w2, w3, w4, w5, [posMV posCVAR posMAD posGINI posGS], LBound, UBound, symbol)

figure;
plot(pRisk1,pRet1,'--b','LineWidth',2)
hold on
plot(pRisk2, pRet2,'r','LineWidth',2)
hold on
plot(pRisk3, pRet3,'y','LineWidth',2)
hold on
plot(pRisk4, pRet4,'g','LineWidth',2)
hold on
plot(pRisk5, pRet5,'c','LineWidth',2)
%title('Efficient Frontiers (Mean-Variance vs CVAR vs MAD vs Mean-Gini vs Golden Section)');
xlabel('Risk of Portfolio');
ylabel('Mean of Portfolio Returns');
legend({'Variance','CVAR','MAD','GMD','GS'},'Location','southeast')


plotGoldenSection(miscGS1,miscGS2,1000)


function AssetScenarios = simEmpirical(ret,nScenario)
    [nSample,nAsset] = size(ret);
    u = zeros(nSample,nAsset);
    for i = 1:nAsset
        u(:,i) = ksdensity(ret(:,i),ret(:,i),'function','cdf');
    end
    [rho, dof] = copulafit('t',u);
    r = copularnd('t',rho,dof,nScenario);
    AssetScenarios = zeros(nScenario,nAsset);
    for i = 1:nAsset
        AssetScenarios(:,i) = ksdensity(ret(:,i),r(:,i),'function','icdf');
    end
end

function plotAssetHist(symbol,ret)
    figure
    nAsset = numel(symbol);
    plotCol = 3;
    plotRow = ceil(nAsset/plotCol);
    for i = 1:nAsset
        subplot(plotRow,plotCol,i);
        histogram(ret(:,i));
        title(symbol{i});
    end
end

function plotWeight(w, symbol, pos, LBound, UBound, title1)
    figure;
    w = round(w'*100,1);
    area(w);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title(title1);
    if pos=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    legend(symbol);
end

function plotWeight2(w1, w2, w3, w4, w5, pos, LBound, UBound, symbol)
    figure;
    subplot(1,5,1)
    w1 = round(w1'*100,1);
    area(w1);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title('Variance');
    xlim([1 10])
    if pos(1,1)=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    %legend(symbol);
    
    subplot(1,5,2)
    w2 = round(w2'*100,1);
    area(w2);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title('CVAR');
    xlim([1 10])
    if pos(1,2)=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    %legend(symbol);
    
    subplot(1,5,3)
    w3 = round(w3'*100,1);
    area(w3);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title('MAD');
    xlim([1 10])
    if pos(1,3)=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    %legend(symbol);
    
    subplot(1,5,4)
    w4 = round(w4'*100,1);
    area(w4);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title('GMD');
    xlim([1 10])
    if pos(1,4)=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    %legend(symbol);
    
    subplot(1,5,5)
    w5 = round(w5'*100,1);
    area(w5);
    ylabel('Portfolio weight (%)')
    xlabel('Port Number')
    title('GS');
    xlim([1 10])
    if pos(1,5)=="+"
        ylim([0 100]);
    else
        ylim([LBound*100 UBound*100]);
    end
    legend(symbol);
end

function plotGoldenSection(misc,miscP,T)
    upTailPareto=misc(1,1);upTailPower=misc(1,2);
    downTailPareto=misc(2,1);downTailPower=misc(2,2);
    upAlpha=misc(3,1);downAlpha=misc(3,2);
    
    upTailParetoP=miscP(1,1);upTailPowerP=miscP(1,2);
    downTailParetoP=miscP(2,1);downTailPowerP=miscP(2,2);
    upAlphaP=miscP(3,1);downAlphaP=miscP(3,2);
    
    phi1=(sqrt(5)+1)/2;phi2=(sqrt(5)-1)/2;
    PowerL = @(x) x.^phi1;ParetoL = @(x) 1-(1-x).^phi2;
    xi = linspace(0,1,T);
    x1=xi(xi<=downAlpha);x2=xi(xi>=downAlpha);x2=x2(x2<=upAlpha);x3=xi(xi>=upAlpha);
    x1P=xi(xi<=downAlphaP);x2P=xi(xi>=downAlphaP);x2P=x2P(x2P<=upAlphaP);x3P=xi(xi>=upAlphaP);
    LZdown = @(x) downTailPower*PowerL(x) + downTailPareto*ParetoL(x);
    LZdownP = @(x) downTailPowerP*PowerL(x) + downTailParetoP*ParetoL(x);
    LZup = @(x) upTailPower*PowerL(x) + upTailPareto*ParetoL(x);
    LZupP = @(x) upTailPowerP*PowerL(x) + upTailParetoP*ParetoL(x);
    y1=LZdown(x1);y3=LZup(x3);
    y1P=LZdownP(x1P);y3P=LZupP(x3P);
    LZcenter = @(x) ((y3(1,1)-y1(1,end))/(x3(1,1)-x1(1,end)))*(x-x3(1,1))+y3(1,1);
    LZcenterP = @(x) ((y3P(1,1)-y1P(1,end))/(x3P(1,1)-x1P(1,end)))*(x-x3P(1,1))+y3P(1,1);
    y2=LZcenter(x2);
    y2P=LZcenterP(x2P);
    
    figure;
    plot(xi,LZdown(xi),'--r','LineWidth',1)
    hold on
    plot(xi,LZdownP(xi),'--r','LineWidth',1)
    hold on
    plot(xi,LZup(xi),':b','LineWidth',1)
    hold on
    plot(xi,LZupP(xi),':b','LineWidth',1)
    hold on
    plot(xi,xi,'g','LineWidth',1)
    hold on
    plot(x1,y1,'Color',1/255*[238 130 238],'LineWidth',2);
    hold on
    plot(x1P,y1P,'Color',1/255*[148 0 211],'LineWidth',2);
    hold on
    y=plot(x3,y3,'Color',1/255*[238 130 238],'LineWidth',2);
    hold on
    y=plot(x3P,y3P,'Color',1/255*[148 0 211],'LineWidth',2);
    hold on
    plot(x2,y2,'Color',1/255*[238 130 238],'LineWidth',2);
    hold on
    plot(x2P,y2P,'Color',1/255*[148 0 211],'LineWidth',2);
    hold on
    scatter(x1(1,end),y1(1,end),'MarkerEdgeColor',1/255*[238 130 238],'MarkerFaceColor', 1/255*[238 130 238],'LineWidth',1)
    hold on
    scatter(x1P(1,end),y1P(1,end),'MarkerEdgeColor',1/255*[148 0 211],'MarkerFaceColor', 1/255*[148 0 211],'LineWidth',1)
    hold on
    scatter(x3(1,1),y3(1,1),'MarkerEdgeColor',1/255*[238 130 238],'MarkerFaceColor',1/255*[238 130 238], 'LineWidth',1)
    ylabel('Optimal portfolio Lorenz curve')
    xlabel('Portfolio return probability')
    grid on
    hold on
    scatter(x3P(1,1),y3P(1,1),'MarkerEdgeColor',1/255*[148 0 211],'MarkerFaceColor',1/255*[148 0 211], 'LineWidth',2)
    ylabel('Optimal portfolio Lorenz curve')
    xlabel('Portfolio return probability')
    grid on
    legend({'Down tail','','Up tail (non-abs. values)','Up tail (abs. values)','Non-stochastic (equality line)','Truncated (non-abs. values)','Truncated (abs. values)'},'Location','southeast')
end