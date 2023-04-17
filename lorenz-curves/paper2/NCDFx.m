function [F] = NCDFx(x,distr,margdistr,copPar1,copPar2,margPar1,margPar2,margPar3,margPar4,margPar5)
% Multivariate CDF based on copula and marginals

% Input:
% x - p x n matrix of the axis grid point; 2=<p<=3
% p - dimension of the CDF, n - dimension of the grid (number of grid points)  
% disrt - copula's family: 'Gaussian', 't', 'Clayton', 'Frank','Gumbel'
% margdistr - marginals family
% copPars - parameters of the copula - 1 or 2 parameters. Leave [] for 
% copPar2 if none. Cases:  
% 'Gaussian' - copPar1: p x p correlation matrix, if p=2 could be a single number
% 't' - copPar1: p x p correlation matrix, copPar2 - degrees of freedom
% 'Clayton', 'Frank','Gumbel' - copPar1 (scalar parameter):
% 'Clayton' - copPar1 in [0,+inf)
% 'Frank' - copPar1 in (-inf,+inf)
% 'Gumbel" - copPar1 in [1,+inf)
% details: 
% https://www.mathworks.com/help/stats/copulacdf.html?searchHighlight=copulacdf&s_tid=srchtitle_copulacdf_1
% marPars - parameters of the marginals - 1,2,...,5 parameters. Leave [] for 
% copPari if none.
% details: 
% https://www.mathworks.com/help/stats/prob.normaldistribution.cdf.html

% Output:
% CDF as a grid
p=length(x(:,1));

switch length(margPar1)
    case 2
        P11=margPar1(1,1);
        P12=margPar1(1,2);
    case 3
        P11=margPar1(1,1);
        P12=margPar1(1,2);
        P13=margPar1(1,3);
    otherwise
        P11=[];
        P12=[];
        P13=[];
end
switch length(margPar2)
    case 2
        P21=margPar2(1,1);
        P22=margPar2(1,2);
    case 3
        P21=margPar2(1,1);
        P22=margPar2(1,2);
        P23=margPar2(1,3);
    otherwise
        P21=[];
        P22=[];
        P23=[];
end
switch length(margPar3)
    case 2
        P31=margPar3(1,1);
        P32=margPar3(1,2);
    case 3
        P31=margPar3(1,1);
        P32=margPar3(1,2);
        P33=margPar3(1,3);
    otherwise
        P31=[];
        P32=[];
        P33=[];
end
switch length(margPar4)
    case 2
        P41=margPar4(1,1);
        P42=margPar4(1,2);
    case 3
        P41=margPar4(1,1);
        P42=margPar4(1,2);
        P43=margPar4(1,3);
    otherwise
        P41=[];
        P42=[];
        P43=[];
end
switch length(margPar5)
    case 2
        P51=margPar5(1,1);
        P52=margPar5(1,2);
    case 3
        P51=margPar5(1,1);
        P52=margPar5(1,2);
        P53=margPar5(1,3);
    otherwise
        P51=[];
        P52=[];
        P53=[];
end

switch p
    case 2
        x1=x{1,:};x2=x{2,:};
        N1=length(x1(1,:));
        N2=length(x2(1,:));
        x11 = cdf(margdistr,x1,P11,P21,P31,P41,P51);
        x22 = cdf(margdistr,x2,P12,P22,P32,P42,P52);
        [X1,X2]=meshgrid(x11,x22);
        FF=copulacdf(distr,[X1(:),X2(:)],copPar1,copPar2);
        F=reshape(FF,N2,N1);
    case 3
        x1=x{1,:};x2=x{2,:};x3=x{3,:};
        N1=length(x1(1,:));
        N2=length(x2(1,:));
        N3=length(x3(1,:));
        x11 = cdf(margdistr,x1,P11,P21,P31,P41,P51);
        x22 = cdf(margdistr,x2,P12,P22,P32,P42,P52);
        x33 = cdf(margdistr,x3,P13,P23,P33,P43,P53);
        [X1,X2,X3]=meshgrid(x11,x22,x33);
        FF=copulacdf(distr,[X1(:),X2(:),X3(:)],copPar1,copPar2);
        F=reshape(FF,N3,N2,N1);
end




