function [x0,x0ind] = NCDFinv(x,y,y0,infx,supx,bound, method)
% Inverse of an empirical distribution function CDF using different
% numerical methods

% Input:
% x - function input array
% y - function output array
% y0 - vector of points at which to find the inverse
% method - the algorithm used:
% 1 - standard numerical inverse through interpolation of the grid
% (monotonicity required)
% 2 - generalized inverse invF(.): invF(y0)=inf(x|F(x)>=y0)

% Output:
% x0 - inverse value
% x0ind - point of grid at for which the inverse occurs (method 2) or is
% closest to the value provided/interpolated (method 1), i.e. in this
% latter instance we return the points as if we use method 2. 

x0=zeros(1,length(y0(1,:)));
x0ind=zeros(1,length(y0(1,:)));

if method==1
    for i=1:1:length(y0)
        [~, ind] = unique(y);
        yy=y(1,ind);xx=x(1,ind);
        x0(1,i)=interp1(yy,xx,y0(1,i));

        %case 1: y0 < min point of the y grid
        % NB: The y grid provided should be dense enough. So it should
        % start close to the inf point of the domain of the inverse
        % function. Yet, if y0 somehow becomes less than min point of the
        % y grid, at this y0 the inverse should be fixed at that inf
        % point of the domain, or due to numerical reasons at the first
        % point of the grid. We assume working (providing) nice grids, so
        % we fix the point at the first point of the grid when dom=0 and
        % we fix the point at infx when dom=1.
        if y0(1,i)<=y(1,1) && bound==0 
            x0ind(1,i)=0; %just encoding (we are outside of x from below)
        end
        if y0(1,i)<=y(1,1) && bound==1 
            x0ind(1,i)=1;
        end  
        if y0(1,i)<=y(1,1) && bound==2 
            x0ind(1,i)=1; %just encoding (we are outside of x from below)
        end
        %case 2: y0 > max point of the y grid
        % NB: Same logic - fix at the max point of the grid or supy
        if y0(1,i)>=y(1,end) && bound==0
            x0ind(1,i)=length(x(1,:))+1; %just encoding (we are outside of
                                         % x from above)
        end
        if y0(1,i)>=y(1,end) && bound==1
            x0ind(1,i)=length(x(1,:));
        end
        if y0(1,i)>=y(1,end) && bound==2
            x0ind(1,i)=x0ind(1,i-1);    % taking inf according to the 
                                        % generalized inverse definition
                                        % and avoiding jumps
        end
        % case 3: y0 within infx and supx
        if  y0(1,i)>y(1,1) && y0(1,i)<y(1,end)
            [k,~] = dsearchn(y',y0(1,i));
            val=y(1,k);
            if val>y0(1,i)
                x0ind(1,i)=k; % the y index is the same as
                              % the x index due to the grid so effectively 
                              % we return an x index
            else
                x0ind(1,i)=k+1;
            end
        end


    end
    mid=length(x0(1,:))/2;
    temp1 = isnan(x0);k1 = find(temp1==1);k2 = find(temp1==0);
    if ~isempty(k1) && ~isempty(k2)
        k11e=k1(1,end);k22e=k2(1,end);
        k11=k1(1,1);k22=k2(1,1);
        % case 1: temp1: 1 1 1 0 ..... 0 0 0 (1-means inverse non-existence)
        if k11e<mid
            x0(1,temp1)=x0(1,k11e+1);
        end
        % case 2: temp1: 0 0 0 ....... 1 1 1 
        if k11>mid
            x0(1,temp1)=x0(1,k11-1);
        end 
        % case 3: temp1: 1 1 1 0 ....0 1 1 1
        if k11<mid && k11e>mid
            mtemp1=temp1(1,1:mid);
            mtemp1e=temp1(1,mid+1:end);
            x0(1,mtemp1)=x0(1,k22);
            x0(1,mtemp1e)=x0(1,k22e);
        end
    end
end

if method==2
    for i=1:1:length(y0)
        %case 1: y0 < min point of the y grid
        % NB: The y grid provided should be dense enough. So it should
        % start close to the inf point of the domain of the inverse
        % function. Yet, if y0 somehow becomes less than min point of the
        % y grid, at this y0 the inverse should be fixed at that inf
        % point of the domain, or due to numerical reasons at the first
        % point of the grid. We assume working (providing) nice grids, so
        % we fix the point at the first point of the grid when dom=0 and
        % we fix the point at infx when dom=1.
        if y0(1,i)<=y(1,1) && bound==0 
            x0(1,i)=infx;
            x0ind(1,i)=0; %just encoding (we are outside of x from below)
        end
        if y0(1,i)<=y(1,1) && bound==1 
            x0(1,i)=x(1,1);
            x0ind(1,i)=1;
        end  
        if y0(1,i)<=y(1,1) && bound==2 
            x0(1,i)=x(1,1);
            x0ind(1,i)=1;
        end
        %case 2: y0 > max point of the y grid
        % NB: Same logic - fix at the max point of the grid or supy
        if y0(1,i)>=y(1,end) && bound==0
            x0(1,i)=supx;
            x0ind(1,i)=length(x(1,:))+1; %just encoding (we are outside of
                                         % x from above)
        end
        if y0(1,i)>=y(1,end) && bound==1
            x0(1,i)=x(1,end);
            x0ind(1,i)=length(x(1,:));
        end
        if y0(1,i)>=y(1,end) && bound==2
            x0ind(1,i)=x0ind(1,i-1);    % taking inf according to the 
                                        % generalized inverse definition
                                        % and avoiding jumps
            x0(1,i)=x(1,i-1);
        end
        % case 3: y0 within infx and supx
        if  y0(1,i)>y(1,1) && y0(1,i)<y(1,end)
            [k,~] = dsearchn(y',y0(1,i));
            val=y(1,k);
            if val>y0(1,i)
                x0ind(1,i)=k; % the y index is the same as
                              % the x index due to the grid so effectively 
                              % we return an x index
                x0(1,i)=x(1,k);
            else
                x0ind(1,i)=k+1;
                x0(1,i)=x(1,k+1);
            end
        end
    end   
end
