clear
% Initializations
%) 1) Initial parameters (Input)
iter=15; % number of iterations
densityX=200; % number of points in [0,1] as a benchmark, x dimension
infx=0;supx=8;% starting distribution domain
bound=0; %treatment of the corner points in the  inverse function method 3
methodInv=1;%the method used for the inverse function calculation on a grid
% 2) Initial distribution F grid  
% grid for x1F
densityXF=(supx-infx)*densityX;
xF = linspace(infx, supx, densityXF+1);
x = linspace(0, 1, densityX+1);
% controlling for possible singularities at the corner points
xF=xF(1,2:end);
x=x(1,2:end);
NF=length(xF(1,:));
N=length(x(1,:));
% starting distribution
P1=0.5; %mean
P2=0.2;%st. dev.
P3=[];
P4=[];
P5=[];
F = cdf('Lognormal',xF,P1,P2,P3,P4,P5);

% Main code

% Lorenz curves initialization (iter - number of iterations, N - number of grid points in [0,1]) 
LZi=zeros(iter,N); % Lorenz curve
ITiRef=zeros(iter,N); % Inverse Lorenz curve - integrated tail method
LZiRef1=zeros(iter,N); % Inverse Lorenz curve - reflected Lorenz curve method

% Lorenz curves inverses initialization (iter - number of iterations, N - number of grid points in [0,1]) 
LZiInv=zeros(iter,N);
LZiInvInd=zeros(iter,N);
ITiRefInv=zeros(iter,N);
phixITi=zeros(iter,N);
ITiRefInvInd=zeros(iter,N);
LZiRefInv1=zeros(iter,N);
LZiRefInvInd1=zeros(iter,N);
LZiRefInvTemp=zeros(iter,N);
temp=zeros(iter,N);

% the initial Lorenz curve
lim=xF(1,end);
% Assuming NCDFinv is a user-defined function or available in the MATLAB path.
% The correct execution of this script depends on your NCDFinv function.
[Finv,FinvInd]=NCDFinv(xF,F,x(1,:),infx,supx,bound,methodInv);
LZiRefInvTemp(1,:)=Finv;
%temp(1,:)=TrimNaN(LZiRefInvTemp(1,:)); % This line was commented out in the original code
temp(1,:)=LZiRefInvTemp(1,:);temp(1,isnan(temp(1,:)))=1;
fxLZ=cumtrapz(x,temp(1,:))/trapz(x,temp(1,:));
fxIT=cumtrapz(xF,1-F)/trapz(xF,1-F);
LZi(1,:)=fxLZ;
phixIT=flip(fxIT(1,:));
[ITiRefInv(1,:), ITiRefInvInd(1,:)]=NCDFinv(xF,phixIT,x(1,:),infx,supx,bound,methodInv);
ITiRef(1,:)=1-ITiRefInv(1,:)/lim;
[LZiRefInv1(1,:), LZiRefInvInd1(1,:)]=NCDFinv(x(1,:),LZi(1,:),1-x(1,:),infx,supx,bound,methodInv);
LZiRef1(1,:)=1-LZiRefInv1(1,:);


% the next Lorenz curves
limi=1;
for i=2:1:iter
    [LZiInv(i-1,:),LZiInvInd(i-1,:)]=NCDFinv(x,LZi(i-1,:),x(1,:),0,1,bound,methodInv);
    LZi(i,:)=cumsum(LZiInv(i-1,:))/sum(LZiInv(i-1,:));

    fxITi=cumtrapz(x,1-ITiRef(i-1,:))/trapz(x,1-ITiRef(i-1,:));
    phixITi(i,:)=flip(fxITi(1,:));
    [ITiRefInv(i,:),ITiRefInvInd(i-1,:)]=NCDFinv(x, phixITi(i,:),x(1,:),0,1,bound,methodInv);
    ITiRef(i,:)=1-ITiRefInv(i,:);
    [LZiRefInv1(i,:), ITiRefInvInd(i-1,:)]=NCDFinv(x(1,:),LZi(i,:),1-x(1,:),0,1,bound,methodInv); % Note: Original code outputs to ITiRefInvInd(i-1,:), not LZiRefInvInd1
    LZiRef1(i,:)=1-LZiRefInv1(i,:);
end

% Define line styles for B&W plots
line_styles = {'-', '--', ':', '-.'};
num_styles = length(line_styles);

figure(1)
legend_entries_fig1 = cell(1, iter-1); % For storing legend text
for i=2:1:iter+1
    current_style_index = mod(i-2, num_styles) + 1; % Cycle through line styles
    plot(x,LZi(i-1,:)', 'LineStyle', line_styles{current_style_index}, 'Color', 'k'); % Plot in black with varying style
    hold on
    legend_entries_fig1{i-1} = sprintf('n=%d', i-1);
end
hold off % Release hold on the plot
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on
legend(legend_entries_fig1, 'Location', 'best'); % Add legend to the plot

figure(2)
legend_entries_fig2 = cell(1, iter-1);
for i=2:1:iter+1
    current_style_index = mod(i-2, num_styles) + 1;
    plot(x,LZiRef1(i-1,:), 'LineStyle', line_styles{current_style_index}, 'Color', 'k'); % Plot in black with varying style
    hold on
    legend_entries_fig2{i-1} = sprintf('n=%d', i-1);
end
hold off
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on
legend(legend_entries_fig2, 'Location', 'best');

figure(3)
legend_entries_fig3 = cell(1, iter-1);
for i=2:1:iter
    current_style_index = mod(i-2, num_styles) + 1;
    plot(x,ITiRef(i,:), 'LineStyle', line_styles{current_style_index}, 'Color', 'k'); % Plot in black with varying style
    hold on
    legend_entries_fig3{i-1} = sprintf('n=%d', i);
end
hold off
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on
legend(legend_entries_fig3, 'Location', 'best');

figure(4)
alpha=(sqrt(5)+1)/2;
beta=1/alpha;
% Plot with distinct black and white line styles and slightly thicker lines
plot(x,x.^alpha, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1.2); 
hold on
plot(x,1-(1-x).^beta, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1.2);
hold off
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on
legend({sprintf('x^{%.2f}', alpha), sprintf('1-(1-x)^{%.2f}', beta)}, 'Location', 'best');






