miscGS1=[0.8 0.2;0.3 0.7;0.75 0.25; 2.5 0];    % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS1
miscGS2=[1 0;0.5 0.5;0.75 0;4 0];            % [upTailPareto upTailPower;downTailPareto downTailPower;upAlpha downAlpha; ni] for GS2

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

    x1=xi(xi<=downAlpha);
    x2=xi(xi>=downAlpha & xi<=upAlpha); 
    x3=xi(xi>=upAlpha);
    
    x1P=xi(xi<=downAlphaP);
    x2P=xi(xi>=downAlphaP & xi<=upAlphaP); 
    x3P=xi(xi>=upAlphaP);

    LZdown_func = @(x) downTailPower*PowerL(x) + downTailPareto*ParetoL(x); % Renamed for clarity
    LZdownP_func = @(x) downTailPowerP*PowerL(x) + downTailParetoP*ParetoL(x); % Renamed
    LZup_func = @(x) upTailPower*PowerL(x) + upTailPareto*ParetoL(x); % Renamed
    LZupP_func = @(x) upTailPowerP*PowerL(x) + upTailParetoP*ParetoL(x); % Renamed
    
    y1=LZdown_func(x1);
    if ~isempty(x3)
        y3=LZup_func(x3);
    else
        y3 = []; 
    end
    
    y1P=LZdownP_func(x1P);
    if ~isempty(x3P)
        y3P=LZupP_func(x3P);
    else
        y3P = [];
    end

    if ~isempty(x1) && ~isempty(x2) && ~isempty(x3) && numel(x1)>0 && numel(x3)>0
        if (x3(1,1)-x1(1,end)) ~= 0
            LZcenter = @(x_val) ((y3(1,1)-y1(1,end))/(x3(1,1)-x1(1,end)))*(x_val-x3(1,1))+y3(1,1);
            y2=LZcenter(x2);
        else 
            if ~isempty(x2) 
                 y2 = y1(1,end) * ones(size(x2)); 
            else
                 y2 = [];
            end
        end
    else
        y2 = []; 
    end

    if ~isempty(x1P) && ~isempty(x2P) && ~isempty(x3P) && numel(x1P)>0 && numel(x3P)>0
        if (x3P(1,1)-x1P(1,end)) ~= 0
            LZcenterP = @(x_val) ((y3P(1,1)-y1P(1,end))/(x3P(1,1)-x1P(1,end)))*(x_val-x3P(1,1))+y3P(1,1);
            y2P=LZcenterP(x2P);
        else 
            if ~isempty(x2P)
                y2P = y1P(1,end) * ones(size(x2P));
            else
                y2P = [];
            end
        end
    else
        y2P = []; 
    end
    
    figure;
    hold on; 

    % Define B&W styles corresponding to original color styles
    % Original '--r' (e.g., LZdown, LZdownP) -> B&W 'k:' (black dotted)
    % Original ':b' (e.g., LZup, LZupP) -> B&W 'k-.' (black dash-dotted)
    % Original 'g' (equality line) -> B&W grey solid
    
    down_tail_style = {'Color', 'k', 'LineStyle', ':', 'LineWidth', 1.2};
    up_tail_style = {'Color', 'k', 'LineStyle', '-.', 'LineWidth', 1.2};

    % 1. 'Down tail' - Corresponds to LZdown_func(xi)
    plot(xi,LZdown_func(xi), down_tail_style{:}, 'DisplayName', 'Down tail');
    % Plot LZdownP_func(xi) with the same style, but NOT in legend
    plot(xi,LZdownP_func(xi), down_tail_style{:}, 'HandleVisibility', 'off');
    
    % 2. 'Up tail (non-abs. values)' - Corresponds to LZup_func(xi)
    plot(xi,LZup_func(xi), up_tail_style{:}, 'DisplayName', 'Up tail (non-abs. values)');
    % 3. 'Up tail (abs. values)' - Corresponds to LZupP_func(xi)
    % LZupP_func(xi) uses the same style as LZup_func(xi) but gets its own legend entry
    plot(xi,LZupP_func(xi), up_tail_style{:}, 'DisplayName', 'Up tail (abs. values)');
    
    % 4. 'Non-stochastic (equality line)'
    plot(xi,xi, 'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 1, 'DisplayName', 'Non-stochastic (equality line)');

    % 5. 'Truncated (non-abs. values)' - GS1 Optimal Lorenz Curve (Thick Solid Black)
    plotted_gs1_optimal = false;
    if ~isempty(x1) && ~isempty(y1)
        plot(x1,y1,'Color','k','LineWidth',2.5, 'LineStyle', '-', 'DisplayName','Truncated (non-abs. values)');
        plotted_gs1_optimal = true;
    end
    if ~isempty(x2) && ~isempty(y2)
        plot(x2,y2,'Color','k','LineWidth',2.5, 'LineStyle', '-', 'HandleVisibility', iff(plotted_gs1_optimal,'off','on'), 'DisplayName', iff(plotted_gs1_optimal,'','Truncated (non-abs. values)'));
        if ~plotted_gs1_optimal, plotted_gs1_optimal = true; end
    end
    if ~isempty(x3) && ~isempty(y3)
        plot(x3,y3,'Color','k','LineWidth',2.5, 'LineStyle', '-', 'HandleVisibility', iff(plotted_gs1_optimal,'off','on'), 'DisplayName', iff(plotted_gs1_optimal,'','Truncated (non-abs. values)'));
    end

    % 6. 'Truncated (abs. values)' - GS2 Optimal Lorenz Curve (Thick Dashed Black)
    plotted_gs2_optimal = false;
    if ~isempty(x1P) && ~isempty(y1P)
        plot(x1P,y1P,'Color','k','LineWidth',2.5, 'LineStyle', '--', 'DisplayName','Truncated (abs. values)');
        plotted_gs2_optimal = true;
    end
    if ~isempty(x2P) && ~isempty(y2P)
        plot(x2P,y2P,'Color','k','LineWidth',2.5, 'LineStyle', '--', 'HandleVisibility', iff(plotted_gs2_optimal,'off','on'), 'DisplayName', iff(plotted_gs2_optimal,'','Truncated (abs. values)'));
        if ~plotted_gs2_optimal, plotted_gs2_optimal = true; end
    end
    if ~isempty(x3P) && ~isempty(y3P)
        plot(x3P,y3P,'Color','k','LineWidth',2.5, 'LineStyle', '--', 'HandleVisibility', iff(plotted_gs2_optimal,'off','on'), 'DisplayName', iff(plotted_gs2_optimal,'','Truncated (abs. values)'));
    end

    % Scatter points - No legend entry
    if ~isempty(x1) && ~isempty(y1) && numel(x1)>0 
        scatter(x1(1,end),y1(1,end),50,'o','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1, 'HandleVisibility','off');
    end
    if ~isempty(x3) && ~isempty(y3) && numel(x3)>0 
        scatter(x3(1,1),y3(1,1),50,'o','MarkerEdgeColor','k','MarkerFaceColor','k', 'LineWidth',1, 'HandleVisibility','off');
    end
    if ~isempty(x1P) && ~isempty(y1P) && numel(x1P)>0
        scatter(x1P(1,end),y1P(1,end),50,'s','MarkerEdgeColor','k','MarkerFaceColor','w','LineWidth',1.2, 'HandleVisibility','off');
    end
    if ~isempty(x3P) && ~isempty(y3P) && numel(x3P)>0
        scatter(x3P(1,1),y3P(1,1),50,'s','MarkerEdgeColor','k','MarkerFaceColor','w', 'LineWidth',1.2, 'HandleVisibility','off');
    end
    
    ylabel('Optimal portfolio Lorenz curve')
    xlabel('Portfolio return probability')
    grid on
    axis([0 1 0 1]); 
    legend_entries = {'Down tail', ...
                      'Up tail (non-abs. values)', ...
                      'Up tail (abs. values)', ...
                      'Non-stochastic (equality line)', ...
                      'Truncated (non-abs. values)', ...
                      'Truncated (abs. values)'};
    legend(legend_entries,'Location','southeast');
    hold off;
end

% Helper function for conditional DisplayName/HandleVisibility
function out = iff(condition, true_val, false_val)
    if condition
        out = true_val;
    else
        out = false_val;
    end
end