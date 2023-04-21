clear; close all; clc; rng(0);
T = readtable('dowPortfolio.xlsx');
symbol = {'AIG','GE','IBM','JPM','MSFT','WMT'};
nAsset = numel(symbol);
ret = tick2ret(T{:,symbol});
plotAssetHist(symbol,ret)
nScenario = 300;
simulationMethod = 'Normal';
switch simulationMethod
    case 'Normal'% Based on normal distribution
        AssetScenarios = mvnrnd(mean(ret),cov(ret),nScenario);
    case 'Empirical' % Based on empirical distribution using t-copula
        AssetScenarios = simEmpirical(ret,nScenario);
end
plotAssetHist(symbol,AssetScenarios)
%CVaR Portfolio Optimization
p1 = PortfolioCVaR('Scenarios', AssetScenarios);
p1 = setDefaultConstraints(p1); 
p1 = setProbabilityLevel(p1, 0.95);
w1 = estimateFrontier(p1,10);
plotWeight(w1, symbol, 'CVaR Portfolio');
portNum = 7; 
plotCVaRHist(p1, w1, ret, portNum, 50)
%MV Portfolio Optimization
p2 = Portfolio;
p2 = setAssetList(p2, symbol);
p2 = estimateAssetMoments(p2, ret);
p2 = setDefaultConstraints(p2);
w2 = estimateFrontier(p2,10);
plotWeight(w2, symbol, 'Mean-Variance Portfolio ');
% Mean-Gini Portfolio Optimization
[w3,retg,riskg] = GiniRisk(AssetScenarios);
plotWeight(w3, symbol, 'Mean-Gini Portfolio ');

pRet1 = estimatePortReturn(p1,w1);
pRisk1 = estimatePortRisk(p1,w1);
pRet2 = estimatePortReturn(p1,w2);
pRisk2 = estimatePortRisk(p1,w2);
pRet3 = estimatePortReturn(p1,w3);
pRisk3 = estimatePortRisk(p1,w3);
plotWeight2(w1, w2, w3, symbol)

figure;
plot(pRisk1,pRet1,'-r','LineWidth',2)
hold on
plot(pRisk2, pRet2,'--b','LineWidth',2)
hold on
plot(pRisk3, pRet3,'g','LineWidth',2)
title('Efficient Frontiers (CVaR vs Mean-Variance vs Mean-Gini)');
xlabel('Risk of Portfolio');
ylabel('Mean of Portfolio Returns');
legend({'CVaR','Mean-Variance','Mean-Gini'},'Location','southeast')

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
figure;
nAsset = numel(symbol);
plotCol = 3;
plotRow = ceil(nAsset/plotCol);
for i = 1:nAsset
    subplot(plotRow,plotCol,i);
    histogram(ret(:,i));
    title(symbol{i});
end
end

function plotCVaRHist(p, w, ret, portNum, nBin)
portRet = ret*w(:,portNum); 
VaR = estimatePortVaR(p,w(:,portNum));
CVaR = estimatePortRisk(p,w(:,portNum));
VaR = -VaR; 
CVaR = -CVaR;
figure;
h1 = histogram(portRet,nBin);
title('Histogram of Returns');
xlabel('Returns')
ylabel('Frequency')
hold on;
edges = h1.BinEdges;
counts = h1.Values.*(edges(1:end-1) < VaR);
h2 = histogram('BinEdges',edges,'BinCounts',counts);
h2.FaceColor = 'r';
plot([CVaR;CVaR],[0;max(h1.BinCounts)*0.80],'--r')
text(edges(1), max(h1.BinCounts)*0.85,['CVaR = ' num2str(round(-CVaR,4))])
hold off;
end

function plotWeight(w, symbol, title1)
figure;
w = round(w'*100,1);
area(w);
ylabel('Portfolio weight (%)')
xlabel('Port Number')
title(title1);
ylim([0 100]);
legend(symbol);
end

function plotWeight2(w1, w2, w3, symbol)
figure;
subplot(1,3,1)
w2 = round(w2'*100,1);
area(w2);
ylabel('Portfolio weight (%)')
xlabel('Port Number')
title('Mean-Variance');
xlim([1 10])
ylim([0 100]);
legend(symbol);

subplot(1,3,2)
w1 = round(w1'*100,1);
area(w1);
ylabel('Portfolio weight (%)')
xlabel('Port Number')
title('CVaR');
xlim([1 10])
ylim([0 100]);
legend(symbol);

subplot(1,3,3)
w3 = round(w3'*100,1);
area(w3);
ylabel('Portfolio weight (%)')
xlabel('Port Number')
title('Mean-Gini');
xlim([1 10])
ylim([0 100]);
legend(symbol);
end