function [F] = TrimNaN(FF)
%FF=Fdensity(:,:,1);
rowsNaN=find(all(isnan(FF),2)); %finding which rows are NaN
colsNaN=find(all(isnan(FF),1)); %finding which columns are NaN
if isempty(rowsNaN)
    lrows=0;
else
    lrows=length(rowsNaN(:,1));
end
if isempty(colsNaN)
    lcols=0;
else
    lcols=length(colsNaN(1,:));
end
if lcols==0 && lrows==0
    F=FF;
else
    if lrows>0
        if lcols>0
            if rowsNaN(end,1)==length(FF(:,1)) && colsNaN(1,end)~=length(FF(1,:))
                FF(end,:)=FF(end-1,:);
            end
        end
        if lcols==0
            if rowsNaN(end,1)==length(FF(:,1))
                FF(end,:)=FF(end-1,:);
            end
        end
    end
    if lcols>0
        if lrows>0
            if (colsNaN(1,end)==length(FF(1,:))) && rowsNaN(end,1)~=length(FF(:,1))
                FF(:,end)=FF(:,end-1);
            end
        end
        if lrows==0
            if colsNaN(1,end)==length(FF(1,:))
                FF(:,end)=FF(:,end-1);
            end
        end
    end
    if lcols>0 && lrows>0
        if colsNaN(1,end)==length(FF(1,:)) && rowsNaN(end,1)==length(FF(:,1))
            FF(:,end)=FF(:,end-1);
            FF(end,:)=FF(end-1,:);
        end
    end


    rowsNaN1=find(all(isnan(FF),2));
    colsNaN1=find(all(isnan(FF),1)); 

    if isempty(rowsNaN1)
        lrows1=0;
    else
        lrows1=length(rowsNaN1(:,1));
    end
    if isempty(colsNaN)
        lcols1=0;
    else
        lcols1=length(colsNaN1(1,:));
    end

    if lcols1==0 && lrows1==0
        F=FF;
    else
        if lrows1>1
            for i=1:1:lrows1
                j=i;
                while j<lrows1 && rowsNaN1(j+1,1)==rowsNaN1(i,1)+j
                    j=j+1;
                end
                tmpr=length(rowsNaN1(i:1:j,1));llr=length(FF(1,:));
                if rowsNaN1(i,1)>1
                    FF(rowsNaN1(i:1:j,1), : ) = ones(tmpr,llr).*FF(rowsNaN1(i,1)-1,:);
                else
                    FF(rowsNaN1(i:1:j,1), : ) = ones(tmpr,llr).*FF(rowsNaN1(j,1)+1,:);
                end
            end
        end
        if lrows1==1 && rowsNaN1(1,1)~=1
            FF(rowsNaN1(1,1),:)=FF(rowsNaN1(1,1)-1,:);
        end
        
        if lcols1>1
            for k=1:1:lcols1
                p=k;
                while p<lcols1 && colsNaN1(1,p+1)==colsNaN1(1,k)+p
                    p=p+1;
                end
                tmpc=length(colsNaN1(:,k:1:p));llc=length(FF(:,1));
                if colsNaN1(:,k)>1
                    FF(:,colsNaN1(:,k:1:p))=ones(llc,tmpc).*FF(:,colsNaN1(:,k)-1);
                else
                    FF(:,colsNaN1(:,k:1:p))=ones(llc,tmpc).*FF(:,colsNaN1(:,p)+1);
                end
            end
        end
        if lcols1==1 && colsNaN1(1,1)~=1
            FF(:,colsNaN1(1,1))=FF(:,colsNaN1(1,1)-1);
        end
        F=FF;
    end
end