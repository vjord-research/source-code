clear
% Initializations
%) 1) Initial parameters (Input)
iter=20; % number of iterations
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
[Finv,FinvInd]=NCDFinv(xF,F,x(1,:),infx,supx,bound,methodInv);
LZiRefInvTemp(1,:)=Finv;
%temp(1,:)=TrimNaN(LZiRefInvTemp(1,:));
temp(1,:)=LZiRefInvTemp(1,:);temp(1,isnan(temp(1,:)))=1;
fxLZ=cumtrapz(x,temp(1,:))/trapz(x,temp(1,:));
fxIT=cumtrapz(xF,1-F)/trapz(xF,1-F);
LZi(1,:)=fxLZ;
phixIT=flip(fxIT(1,:));
[ITiRefInv(1,:), ITiRefInvInd(1,:)]=NCDFinv(xF,phixIT,x(1,:),infx,supx,bound,methodInv);
ITiRef(1,:)=1-ITiRefInv(1,:)/lim;
[LZiRefInv1(1,:), LZiRefInvInd1(1,:)]=NCDFinv(x(1,:),LZi(1,:),1-x(1,:),infx,supx,bound,methodInv);
LZiRef1(1,:)=1-LZiRefInv1(1,:);

figure(1)
tiledlayout(3,1)
ax1 = nexttile;
plot(ax1,x,LZi(1,:)');
title(ax1,'Lorenz curve - first iteration')
ax2 = nexttile;
plot(ax2,x,ITiRef(1,:));
title(ax2,'Reflected Lorenz curve (IT method) - first iteration')
ax3 = nexttile;
plot(ax3,x,LZiRef1(1,:));
title(ax3,'Reflected Lorenz curve (s. refl. method) - first iteration')
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

% the next Lorenz curves
limi=1;
for i=2:1:iter
    [LZiInv(i-1,:),LZiInvInd(i-1,:)]=NCDFinv(x,LZi(i-1,:),x(1,:),0,1,bound,methodInv);
    LZi(i,:)=cumsum(LZiInv(i-1,:))/sum(LZiInv(i-1,:));

    fxITi=cumtrapz(x,1-ITiRef(i-1,:))/trapz(x,1-ITiRef(i-1,:));
    phixITi(i,:)=flip(fxITi(1,:));
    [ITiRefInv(i,:),ITiRefInvInd(i-1,:)]=NCDFinv(x, phixITi(i,:),x(1,:),0,1,bound,methodInv);
    ITiRef(i,:)=1-ITiRefInv(i,:);
    [LZiRefInv1(i,:), ITiRefInvInd(i-1,:)]=NCDFinv(x(1,:),LZi(i,:),1-x(1,:),0,1,bound,methodInv);
    LZiRef1(i,:)=1-LZiRefInv1(i,:);
end

figure(2)
for i=2:1:iter
    plot(x,LZi(i,:)');
    title('Lorenz curve - iterations')
    hold on
end
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(3)
for i=2:1:iter
    plot(x,ITiRef(i,:));
    title('Reflected Lorenz curve (IT method) - iterations')
    hold on
end
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(4)
for i=2:1:iter
    plot(x,LZiRef1(i,:));
    title('Reflected Lorenz curve (s. refl. method) - iterations')
    hold on
end
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(5)
plot(x,LZi(end,:));
title('Joint limits: Primal LZ, Reflected LZ (IT method), and Reflected LZ (s. refl. method)')
hold on
plot(x,ITiRef(end,:));
hold on
plot(x,LZiRef1(end,:));
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(6)
alpha=(sqrt(5)+1)/2;
beta=1/alpha;
plot(x,x.^alpha);
title('Lorenz curve theoretical limit - joint: primal and reflected')
hold on
plot(x,1-(1-x).^beta);
axis([0 1 0 1])
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(7)
plot(x,LZi(iter,:)-x.^alpha);
title('Error: Lorenz curve - Power Law')
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(8)
plot(x,ITiRef(iter,:)-(1-(1-x).^beta));
title('Error: Reflected Lorenz curve (IT method) - Pareto Law')
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on

figure(9)
plot(x,LZiRef1(iter,:)-(1-(1-x).^beta));
title('Error: Reflected Lorenz curve (s. refl. method) - Pareto Law')
ylabel('Lorenz curve')
xlabel('Cumulative probability')
grid on





