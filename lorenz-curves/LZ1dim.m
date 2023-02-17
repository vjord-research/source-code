clear
% Initializations
%) 1) Initial parameters (Input)
iter=70; % number of iterations
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
LZi=zeros(iter,N);
ITiRef=zeros(iter,N);
LZiRef1=zeros(iter,N);
ITi=zeros(iter,NF);
LZiRef=zeros(iter,N);
LZiDual=zeros(iter,N);

% Lorenz curves inverses initialization (iter - number of iterations, N - number of grid points in [0,1]) 
LZiInv=zeros(iter,N);
LZiDualInv=zeros(iter,N);
LZiDualInvInd=zeros(iter,N);
LZiInvInd=zeros(iter,N);
ITiRefInv=zeros(iter,N);
phixITi=zeros(iter,N);
phixLZi=zeros(iter,N);
phixdLZi=zeros(iter,N);
ITiRefInvInd=zeros(iter,N);
LZiRefInv1=zeros(iter,N);
LZiRefInvInd1=zeros(iter,N);
LZiRefInv=zeros(iter,N);
LZiRefInvTemp=zeros(iter,N);
LZiRefInvInd=zeros(iter,N);
LZiRefInvIndTemp=zeros(iter,N);
temp=zeros(iter,N);
tempd=zeros(iter,N);

% the initial Lorenz curve
lim=xF(1,end);
[Finv,FinvInd]=NCDFinv(xF,F,x(1,:),bound,methodInv);
LZiRefInvTemp(1,:)=Finv;
%temp(1,:)=TrimNaN(LZiRefInvTemp(1,:));
temp(1,:)=LZiRefInvTemp(1,:);temp(1,isnan(temp(1,:)))=1;
LZiRefInvIndTemp(1,:)=FinvInd;
fxLZ=cumtrapz(x,temp(1,:))/trapz(x,temp(1,:));
fxIT=cumtrapz(xF,1-F)/trapz(xF,1-F);
LZi(1,:)=fxLZ;
phixIT=flip(fxIT(1,:));
phixLZ=flip(fxLZ(1,:));
LZiDual(1,:)=1-phixLZ;
[ITiRefInv(1,:), ITiRefInvInd(1,:)]=NCDFinv(xF,phixIT,x(1,:),bound,methodInv);
ITiRef(1,:)=1-ITiRefInv(1,:)/lim;
[LZiRefInv1(1,:), LZiRefInvInd1(1,:)]=NCDFinv(x(1,:),LZi(1,:),1-x(1,:),bound,methodInv);
LZiRef1(1,:)=1-LZiRefInv1(1,:);
ITi(1,:)=fxIT;
[LZiRefInv(1,:), LZiRefInvInd(1,:)]=NCDFinv(x,phixLZ,x(1,:),bound,methodInv);
LZiRef(1,:)=1-LZiRefInv(1,:);

figure(1)
tiledlayout(4,1)
ax1 = nexttile;
plot(ax1,x,[LZi(1,:)' LZiDual(1,:)']);
title(ax1,'Primal/Dual Lorenz curve - first iteration')
ax2 = nexttile;
plot(ax2,x,ITiRef(1,:));
title(ax2,'Inverse Integrated Tail (reflected) curve - first iteration')
ax3 = nexttile;
plot(ax3,x,LZiRef1(1,:));
title(ax3,'Inverse Lorenz curve (reflected method) - first iteration')
ax4 = nexttile;
plot(ax4,xF,ITi(1,:));
title(ax4,'Integrated tail curve - first iteration')

% the next Lorenz curves
limi=1;
for i=2:1:iter
    [LZiInv(i-1,:),LZiInvInd(i-1,:)]=NCDFinv(x,LZi(i-1,:),x(1,:),bound,methodInv);
    [LZiRefInvTemp(i,:),LZiRefInvIndTemp(i,:)]=NCDFinv(x,LZiRef(i-1,:),x(1,:),bound,methodInv);
    [LZiDualInv(i,:),LZiDualInvInd(i,:)]=NCDFinv(x,LZiDual(i-1,:),x(1,:),bound,methodInv);
    LZi(i,:)=cumsum(LZiInv(i-1,:))/sum(LZiInv(i-1,:));
    fxITi=cumtrapz(x,1-ITiRef(i-1,:))/trapz(x,1-ITiRef(i-1,:));
    temp(i,:)=LZiRefInvTemp(i,:);temp(i,isnan(temp(i,:)))=1;
    tempd(i,:)=LZiDualInv(i,:);tempd(i,isnan(tempd(i,:)))=1;
    fxLZi=cumtrapz(x,temp(i,:))/trapz(x,temp(i,:));
    fxdLZi=cumtrapz(x,tempd(i,:))/trapz(x,tempd(i,:));
    phixITi(i,:)=flip(fxITi(1,:));
    phixLZi(i,:)=flip(fxLZi(1,:));
    phixdLZi(i,:)=flip(fxdLZi(1,:));
    [ITiRefInv(i,:),ITiRefInvInd(i-1,:)]=NCDFinv(x, phixITi(i,:),x(1,:),bound,methodInv);
    ITiRef(i,:)=1-ITiRefInv(i,:);
    [LZiRefInv1(i,:), ITiRefInvInd(i-1,:)]=NCDFinv(x(1,:),LZi(i,:),1-x(1,:),bound,methodInv);
    LZiRef1(i,:)=1-LZiRefInv1(i,:);
    ITi(i,:)=cumtrapz(xF,1-ITi(i-1,:))/trapz(xF,1-ITi(i-1,:));
    [LZiRefInv(i,:), LZiRefInvInd(i,:)]=NCDFinv(x,phixLZi(i,:),x(1,:),bound,methodInv);
    LZiRef(i,:)=1-LZiRefInv(i,:);
    LZiDual(i,:)=1-phixdLZi(i,:);
end

figure(2)
for i=2:1:iter
    plot(x,[LZi(i,:)' LZiDual(i,:)']);
    title('Primal/Dual Lorenz curve - iterations')
    hold on
end

figure(3)
for i=2:1:iter
    plot(x,ITiRef(i,:));
    title('Inverse Integrated Tail (reflected) curve - iterations')
    hold on
end

figure(4)
for i=2:1:iter
    plot(x,LZiRef1(i,:));
    title('Inverse Lorenz curve (refl. method) - iterations')
    hold on
end

figure(5)
plot(x,LZi(end,:));
title('Joint limits: primal LZ, inverse IT, and LZ (refl. method)')
hold on
plot(x,ITiRef(end,:));
hold on
plot(x,LZiRef1(end,:));

figure(6)
alpha=(sqrt(5)+1)/2;
beta=1/alpha;
plot(x,x.^alpha);
title('Lorenz curve theoretical limit - joint: primal and inverse')
hold on
plot(x,1-(1-x).^beta);

figure(7)
plot(x,LZi(iter,:)-x.^alpha);
title('Error: Lorenz curve - Power Law')

figure(8)
plot(x,ITiRef(iter,:)-(1-(1-x).^beta));
title('Error: Integrated Tail curve inverse - Power Law')

figure(9)
plot(x,LZiRef1(iter,:)-(1-(1-x).^beta));
title('Error: Lorenz curve inverse (refl. method) - Power Law')





