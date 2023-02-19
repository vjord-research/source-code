function [F] = Fval(X1,X2,Y,X1val,X2val,X1valInd,X2valInd,method)
    if method==1
        FF = interp2(X1,X2,Y,X1val,X2val);
        F=TrimNaN(FF);
    end
    if method==2
        F = Y(X2valInd,X1valInd);
    end
end