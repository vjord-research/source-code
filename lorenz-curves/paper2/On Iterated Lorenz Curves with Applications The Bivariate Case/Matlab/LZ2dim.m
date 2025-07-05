clear
% Initializations
%) 1) Initial parameters (Input)
iter=20; % number of iterations
densityX1=100; % number of points in [0,1] as a benchmark, x1 dimension
densityX2=200; % number of points in [0,1] as a benchmark, x2 dimension
infx=0;supx=8;% starting distribution domain
boundF=2; %treatment of the corner points in the  inverse function method 2
boundi=1;
methodInvF=1;%the method used for the inverse function calculation on a grid
methodInvi=1;
methodFValF=1;%the method used for the function value calculation on a grid
methodFVali=1;
trimNaNind=1;%to trim or not any NaN values that could appear
densityXF1=(supx-infx)*densityX1; % density for the initial distribution
densityXF2=(supx-infx)*densityX2; % density for the initial distribution
% 2) Initial distribution F grid  
% grid for x1F, x2F
x1F = linspace(infx, supx, densityXF1+1);
x2F = linspace(infx, supx, densityXF2+1);
% controlling for possible singularities at the corner points
x1F=x1F(1,2:end);
x2F=x2F(1,2:end);
[X1F,X2F] = meshgrid(x1F,x2F);
% 3) Initial distribution F
% marginals
margdistr='LogNormal';
mPar1=[0.2 0.7]; % mean in case of normal, lognormal, etc.
mPar2=[0.4 0.6]; % st.dev in case of normal, lognormal, etc.
mPar3=[];
mPar4=[];
mPar5=[];
% copula
distr='Gaussian';
copPar1=0; %correlation or other
copPar2=[];

F=NCDFx({x1F;x2F},distr,margdistr,copPar1,copPar2,mPar1,mPar2,mPar3,mPar4,mPar5);
Fdensity=NPDFx({x1F;x2F},distr,margdistr,copPar1,copPar2,mPar1,mPar2,mPar3,mPar4,mPar5);

% 4) Lorenz curves
% consequent Lorenz curves grid, now on [0,1] domain
x1 = linspace(0, 1, densityX1+1);
x2 = linspace(0, 1, densityX2+1);
% controlling for possible singularities at the corner points
x1=x1(1,2:end);
x2=x2(1,2:end);
[X1,X2] = meshgrid(x1,x2);

% The main code
% dimension
N1=length(x1);N2=length(x2);
% marginal cdfs
Fx2=F(:,end)';Fx1=F(end,:);
% marginal cdfs inverses
[Fx1inv,Fx1invInd]=NCDFinv(x1F,Fx1,x1(1,:),infx,supx,boundF,methodInvF);
[Fx2inv,Fx2invInd]=NCDFinv(x2F,Fx2,x2(1,:),infx,supx,boundF,methodInvF);
% Lorenz cdfs and marginal cdfs initialization (iter - number of iterations, N - number of grid points in [0,1])
LZx1i=zeros(iter,N1);LZx1ii=zeros(iter,N1); % LZx1i - Lorenz curve, LZx1ii - Simple Lorenz curve (the one from the independence copula, 1D proxy
LZx2i=zeros(iter,N2);LZx2ii=zeros(iter,N2);
LZi=zeros(N2,N1,iter);
% auxiliary variables initialization
LZiTemp=zeros(N2,N1,iter);prodTemp=zeros(N2,N1,iter);prodTemp1=zeros(N2,N1,iter);
SpearmanRhoi=zeros(iter,1);
corrproxi=zeros(iter,1);
corrprox1i=zeros(N2,N1,iter);
tempx1=zeros(iter,N1);tempx2=zeros(iter,N2);
tempx1Ind=zeros(iter,N1);tempx2Ind=zeros(iter,N2);
tempx11=zeros(iter,N1);tempx22=zeros(iter,N2);
ExpLZi=zeros(iter,1);
ExpLZi1=zeros(iter,1);
ExpLZx1i=zeros(iter,1);ExpLZx1ii=zeros(iter,1);ExpLZx1i1=zeros(iter,1);ExpLZx1i11=zeros(iter,1);
ExpLZx2i=zeros(iter,1);ExpLZx2ii=zeros(iter,1);ExpLZx2i2=zeros(iter,1);ExpLZx2i22=zeros(iter,1);
func22=zeros(N1,N2);
func22i=zeros(N1,N2);
% marginal Lorenz cdfs inverses initialization (iter - number of iterations, N - number of grid points in [0,1])
LZx1invi=zeros(iter,N1);LZx1inviInd=zeros(iter,N1);
LZx2invi=zeros(iter,N2);LZx2inviInd=zeros(iter,N2);
% marginal simple Lorenz cdfs inverses initialization (iter - number of iterations, N - number of grid points in [0,1])
LZx1invii=zeros(iter,N1);LZx1inviiInd=zeros(iter,N1);
LZx2invii=zeros(iter,N2);LZx2inviiInd=zeros(iter,N2);
% auxiliary variables
an=zeros(iter,1);bn=zeros(iter,1);
an(1,1)=1;bn(1,1)=1;
% Lorenz curve first iteration
if trimNaNind==1
    s1temp = Fx1inv(1,:);s2temp = Fx2inv(1,:);
    s1=TrimNaN(s1temp);s2=TrimNaN(s2temp);
    s1Ind = Fx1invInd(1,:);s2Ind = Fx2invInd(1,:);
    [S1,S2] = meshgrid(s1,s2);
    FvTemp=Fval(X1F,X2F,F,S1,S2,s1Ind,s2Ind,methodFValF);
    FvTemp1=Fval(X1F,X2F,F,S1,S2,s1Ind,s2Ind,2);
    Fv=TrimNaN(FvTemp);
    Fv1=TrimNaN(FvTemp1);
    prodTemp=S1.*S2.*Fv;
    prod=TrimNaN(prodTemp);

    prod1=S1.*S2.*X1.*X2;
    prodTemp(:,:,1)=prod./prod1;
else
    s1 = Fx1inv(1,:);s2 = Fx2inv(1,:);
    s1Ind = Fx1invInd(1,:);s2Ind = Fx2invInd(1,:);
    [S1,S2] = meshgrid(s1,s2);
    Fv=Fval(X1F,X2F,F,S1,S2,s1Ind,s2Ind,methodFValF);
    Fv1=Fval(X1F,X2F,F,S1,S2,s1Ind,s2Ind,2);
    prod=S1.*S2.*Fv;

    prod1=S1.*S2.*X1.*X2;
    prodTemp(:,:,1)=prod./prod1;
end

func1=cumtrapz(s2,Fv',2);func1rho=trapz(x2,Fv',2);

func1aa=cumtrapz(s2,Fv',2);func1a=S1.*func1aa';
func1bb=cumtrapz(s1,Fv,2);func1b=S2.*func1bb;

if length(s1)==1
    func2=0;
end
if length(s1)>1
    func2 = cumtrapz(s1,func1); 
end
if length(x1)==1
    func1rho=0;
end
if length(x1)>1
    func2rho = trapz(x1,func1rho); 
end

temp=prod+func2'-func1a-func1b; %integration by parts of a Stieltjes integral
LZiTemp(:,:,1)=temp;
tempp=LZiTemp(:,:,1)./LZiTemp(end,end,1);
LZi(:,:,1)=tempp;
SpearmanRhoi(1,1)=12*func2rho-3;
ExpLZF=LZiTemp(end,end,1);
ExpLZF1=1-trapz(s1,trapz(s2,LZi(:,:,1)',2));

% marginal Lorenz curves first iteration (iter - number of iterations, N - number of grid points in [0,1]) 
LZx1i(1,:)=LZi(end,:,1);LZx2i(1,:)=LZi(:,end,1)';
%ExpLZx1F=trapz(x1,1-LZx1i(1,:));ExpLZx2F=trapz(x2,1-LZx2i(1,:));
ExpLZx1F=trapz(x1F,1-Fx1(1,:));ExpLZx2F=trapz(x2F,1-Fx2(1,:));
corrproxi(1,1)=ExpLZF/(ExpLZx1F*ExpLZx2F);
[LZx1iT,LZx2iT]=meshgrid(LZx1i(1,:),LZx2i(1,:));
corrprox1i(:,:,1)=LZi(:,:,1)./(LZx1iT.*LZx2iT);

% marginal simple Lorenz curves first iteration (iter - number of iterations, N - number of grid points in [0,1]) 
tempFx1=s1;tempFx1(1,isnan(tempFx1(1,:)))=1;
tempFx2=s2;tempFx2(1,isnan(tempFx2(1,:)))=1;
tempFx1Ind=s1Ind;tempFx1Ind(1,isnan(tempFx1Ind(1,:)))=1;
tempFx2Ind=s2Ind;tempFx2Ind(1,isnan(tempFx2Ind(1,:)))=1;
ExpLZFx1ii=trapz(x1,tempFx1);ExpLZFx2ii=trapz(x2,tempFx2);
LZx1ii(1,:)=cumtrapz(x1,tempFx1)/ExpLZFx1ii;
LZx2ii(1,:)=cumtrapz(x2,tempFx2)/ExpLZFx2ii;

func22temp1=cumtrapz(tempFx1,F(end,tempFx1Ind));
func22temp2=cumtrapz(tempFx2,F(tempFx2Ind,end)');
[Func22temp1, Func22temp2]=meshgrid(func22temp1, func22temp2);
prodTemp1(:,:,1)=func2'./(Func22temp1.*Func22temp2);

% the initial distribution pdf and cdf
figure(1)
surf(X1F,X2F,TrimNaN(Fdensity(:,:,1)))
shadowplot x   
shadowplot y   
title('Initial distribution multivariate PDF')

figure(2)
surf(X1F,X2F,F(:,:,1))
title('Initial distribution multivariate CDF')

% the initial distribution marginals cdfs
figure(3)
tiledlayout(2,1)

ax1F = nexttile;
plot(ax1F,x1F,Fx1(1,:));
title(ax1F,'Initial distribution marginal CDF x1')

ax2F = nexttile;
plot(ax2F,x2F,Fx2(1,:));
title(ax2F,'Initial distribution marginal CDF x2')

% the initial marginal Lorenz curves
figure(4)
tiledlayout(2,1)

ax1 = nexttile;
plot(ax1,x1,LZx1i(1,:));
title(ax1,'Marginal Lorenz curve x1 - 1st iteration')

ax2 = nexttile;
plot(ax2,x2,LZx2i(1,:));
title(ax2,'Marginal Lorenz curve x2 - 1st iteration')

figure(5)
tiledlayout(2,1)

ax1s = nexttile;
plot(ax1s,x1,LZx1ii(1,:));
title(ax1s,'Marginal Simple Lorenz curve x1 - 1st iteration')

ax2s = nexttile;
plot(ax2s,x2,LZx2ii(1,:));
title(ax2s,'Marginal Simple Lorenz curve x2 - 1st iteration')

% the initial Lorenz curve
figure(6);
surf(X1,X2,LZi(:,:,1))
title('Initial Lorenz curve')


% the next Lorenz curves
for i=2:1:iter
    [LZx1invi(i-1,:),LZx1inviInd(i-1,:)]=NCDFinv(x1,LZx1i(i-1,:),x1(1,:),infx,supx,boundi,methodInvi);
    [LZx2invi(i-1,:),LZx2inviInd(i-1,:)]=NCDFinv(x2,LZx2i(i-1,:),x2(1,:),infx,supx,boundi,methodInvi); 
    [LZx1invii(i-1,:),LZx1inviiInd(i-1,:)]=NCDFinv(x1,LZx1ii(i-1,:),x1(1,:),infx,supx,boundi,methodInvi);
    [LZx2invii(i-1,:),LZx2inviiInd(i-1,:)]=NCDFinv(x2,LZx2ii(i-1,:),x2(1,:),infx,supx,boundi,methodInvi);
    if trimNaNind==1
        s1iTemp = LZx1invi(i-1,:);s2iTemp = LZx2invi(i-1,:);
        s1i=TrimNaN(s1iTemp);s2i=TrimNaN(s2iTemp);
        s1iiTemp = LZx1invii(i-1,:);s2iiTemp = LZx2invii(i-1,:);
        s1ii=TrimNaN(s1iiTemp);s2ii=TrimNaN(s2iiTemp);
        s1iInd = LZx1inviInd(i-1,:);s2iInd = LZx2inviInd(i-1,:);
        [S1i,S2i] = meshgrid(s1i,s2i);
        FviTempi=Fval(X1,X2,LZi(:,:,i-1),S1i,S2i,s1iInd,s2iInd,methodFVali);
        Fvi=TrimNaN(FviTempi);
        prodiTemp=S1i.*S2i.*Fvi;
        prodi=TrimNaN(prodiTemp);
        FviTempi1=Fval(X1,X2,LZi(:,:,i-1),S1i,S2i,s1iInd,s2iInd,2);
        Fvi1=TrimNaN(FviTempi1);

        prodi1=S1i.*S2i.*X1.*X2;
        prodTemp(:,:,i)=prodi./prodi1;
    else
        s1i = LZx1invi(i-1,:);s2i = LZx2invi(i-1,:);
        s1ii = LZx1invii(i-1,:);s2ii = LZx2invii(i-1,:);
        s1iInd = LZx1inviInd(i-1,:);s2iInd = LZx2inviInd(i-1,:);
        [S1i,S2i] = meshgrid(s1i,s2i);
        Fvi=Fval(X1,X2,LZi(:,:,i-1),S1i,S2i,s1iInd,s2iInd,methodFVali);
        prodi=S1i.*S2i.*Fvi;

        prodi1=S1i.*S2i.*X1.*X2;
        prodTemp(:,:,i)=prodi./prodi1;
        Fvi1=Fval(X1,X2,LZi(:,:,i-1),S1i,S2i,s1iInd,s2iInd,2);
    end
   
    func1i=cumtrapz(s2i,Fvi',2);func1rhoi=trapz(x2,Fvi',2);
    func1aai=cumtrapz(s2i,Fvi',2);func1ai=S1i.*func1aai';
    func1bbi=cumtrapz(s1i,Fvi,2);func1bi=S2i.*func1bbi;
    if length(s1i)==1
        func2i=0;
    end
    if length(s1i)>1
        func2i = cumtrapz(s1i,func1i);
    end
    if length(x1)==1
        func2rhoi=0;
    end
    if length(x1)>1
        func2rhoi = trapz(x1,func1rhoi);
    end

    tempi=prodi+func2i'-func1ai-func1bi; %integration by parts of a Stieltjes integral 

    LZiTemp(:,:,i)=tempi;   
    tempii=LZiTemp(:,:,i)./LZiTemp(end,end,i);
    LZi(:,:,i)=tempii;
    LZx1i(i,:)=LZi(end,:,i);LZx2i(i,:)=LZi(:,end,i)';
    SpearmanRhoi(i,1)=12*func2rhoi-3;
    ExpLZi(i-1,1)=LZiTemp(end,end,i);
    ExpLZi1(i-1,1)=1-trapz(s1i,trapz(s2i,Fvi1',2));

    tempx11(i-1,:)=s1ii;tempx11(1,isnan(tempx11(i-1,:)))=1;
    tempx22(i-1,:)=s2ii;tempx22(1,isnan(tempx22(i-1,:)))=1;
    ExpLZx1ii(i-1,1)=trapz(x1,tempx11(i-1,:));ExpLZx2ii(i-1,1)=trapz(x2,tempx22(i-1,:));
    LZx1ii(i,:)=cumtrapz(x1,tempx11(i-1,:))/ExpLZx1ii(i-1,1);
    LZx2ii(i,:)=cumtrapz(x2,tempx22(i-1,:))/ExpLZx2ii(i-1,1);

    tempx1(i-1,:)=s1i;tempx1(1,isnan(tempx1(i-1,:)))=1;
    tempx2(i-1,:)=s2i;tempx2(1,isnan(tempx2(i-1,:)))=1;
    tempx1Ind(i-1,:)=s1iInd;tempx1Ind(1,isnan(tempx1Ind(i-1,:)))=1;
    tempx2Ind(i-1,:)=s2iInd;tempx2Ind(1,isnan(tempx2Ind(i-1,:)))=1;
    ExpLZx1i(i-1,1)=trapz(x1,1-LZx1i(i-1,:));ExpLZx2i(i-1,1)=trapz(x2,1-LZx2i(i-1,:));
    ExpLZx1i1(i-1,1)=trapz(x1,s1i);ExpLZx2i2(i-1,1)=trapz(x2,s2i);
    ExpLZx1i11(i-1,1)=trapz(x1,s1i);ExpLZx2i22(i-1,1)=trapz(x2,s2i);
    corrproxi(i,1)=ExpLZi(i-1,1)/(ExpLZx1i(i-1,1)*ExpLZx2i(i-1,1));
    [LZx1iT,LZx2iT]=meshgrid(LZx1i(i-1,:),LZx2i(i-1,:));
    corrprox1i(:,:,i)=LZi(:,:,i-1)./(LZx1iT.*LZx2iT);

    func22temp1i=cumtrapz(tempx1(i-1,:),LZx1i(i-1,tempx1Ind(i-1,:)));
    func22temp2i=cumtrapz(tempx2(i-1,:),LZx2i(i-1,tempx2Ind(i-1,:)));
    [Func22temp1i, Func22temp2i]=meshgrid(func22temp1i, func22temp2i);
    prodTemp1(:,:,i)=func2i'./(Func22temp1i.*Func22temp2i);
   
    if i==2
        an(i,1)=1;bn(i,1)=1;
    end
    if i>=3
        an(i,1)=1+1/an(i-1,1);
        bn(i,1)=1/an(i,1);
    end
end

figure(7)
for i=2:1:iter
    plot(x1,LZx1i(i,:));
    hold on
    title('Lorenz curve x1 - iterations')
end
figure(8)
for i=2:1:iter
    plot(x2,LZx2i(i,:));
    hold on
    title('Lorenz curve x2 - iterations')
end

figure(9)
for i=2:1:iter
    plot(x1,LZx1ii(i,:));
    hold on
    title('Simple Lorenz curve x1 - iterations (independence copula)')
end
figure(10)
for i=2:1:iter
     plot(x2,LZx2ii(i,:));
     hold on
    title('Simple Lorenz curve x2 - iterations (independence copula)')
end

figure(11)
for i=1:1:iter
    surf(X1,X2,LZi(:,:,i))
    shadowplot x   
    shadowplot y
    hold on
    title('Lorenz curve 3D - iterations')
end

figure(12)
tiledlayout(2,1)
alpha=(sqrt(5)+1)/2;
x1a=x1.^alpha;x2a=x2.^alpha;
errorLPowerLawx1=LZx1i(iter,:)-x1a;
errorLPowerLawx2=LZx2i(iter,:)-x2a;

ax1er = nexttile;
plot(ax1er,x1,errorLPowerLawx1);
title(ax1er,'Error: Lorenz curve - Power Law x1')

ax2er = nexttile;
plot(ax2er,x2,errorLPowerLawx2);
title(ax2er,'Error: Lorenz curve - Power Law x2')

figure(13)
alpha=(sqrt(5)+1)/2;
Xa=(X1.^alpha).*(X2.^alpha);
surf(X1,X2,Xa)
title('Lorenz curve theoretical limit')

figure(14)
errorLPowerLaw=LZi(:,:,iter)-Xa;
surf(X1,X2,errorLPowerLaw)
title('Error: Lorenz curve - Power Law 2D')

figure(15)
plot(1:1:iter-1,ExpLZi(1:iter-1,1));
title('2D expectation through iterations')

figure(16)
plot(1:1:iter,SpearmanRhoi(:,1));
title('Spearman Rho through iterations')

figure(17)
plot(1:1:iter-1,corrproxi(1:iter-1,1));
title('Check1: Idependence proxy through iterations')

figure(18)
for i=iter
    surf(X1,X2,corrprox1i(:,:,i))
    shadowplot x   
    shadowplot y
    hold on
    title('Check2: Idependence proxy through iterations')
end

figure(19)
for i=iter
    surf(X1,X2,prodTemp(:,:,i))
    shadowplot x   
    shadowplot y
    hold on
    title('Check3: Idependence proxy through iterations')
end

figure(20)
for i=iter
    surf(X1,X2,prodTemp1(:,:,i))
    shadowplot x   
    shadowplot y
    hold on
    title('Check4: Idependence proxy through iterations')
end


