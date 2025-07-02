miscGS1=[0.8 0.2;0.3 0.7;0.75 0.25; 2.5 0];    % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS1. Must be set even if not used due to a plot to be generated.
miscGS2=[1 0;0.5 0.5;0.75 0;4 0];            % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS2. Must be set even if not used due to a plot to be generated.

plotGoldenSection(miscGS1,miscGS2,1000)

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