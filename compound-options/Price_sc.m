function [P_s]=Price_sc(V,s,Vbar,r,vol,t0,t,alpha,beta,init,MaxStep,StepSize)
% this function computes the price and the hedging instruments
% characteristics
rand('seed',0);
nn=length(t);

hh=zeros(nn,1);
tt=zeros(nn,1);
alphaa=zeros(nn,1);
betaa=zeros(nn,1);
VbarNew=zeros(nn,1);

Price_si=0;
prod1=1;prod2=1;sum1=0;

s1=nn;

for jj=nn:-1:s    
    hh(nn-jj+1,1)=t(s+nn-jj,1)-t0;
    tt(nn-jj+1,1)=t(s+nn-jj,1);
    alphaa(nn-jj+1,1)=alpha(s+nn-jj,1);
    betaa(nn-jj+1,1)=beta(s+nn-jj,1);
    VbarNew(nn-jj+1,1)=Vbar(s+nn-jj,1); 
    prod1=prod1*alpha(s+nn-jj,1);
    if jj~=s        
        prod2=prod2*alpha(s+nn-jj,1);
        sum1=sum1+prod2*beta(s+nn-jj+1,1)*exp(-(t(s+nn-jj+1,1)-t0));
    end
end
resid=V*prod1+sum1;
Price_j=0;
a=1;

for j=0:1:nn-s
    perm = combnk([nn-s+1:-1:1],nn-s+1-j)';     
    t_i=zeros(length(perm(:,1)),length(perm(1,:)));
    K_i=zeros(length(perm(:,1)),length(perm(1,:)));
    alpha_i=zeros(length(perm(:,1)),length(perm(1,:)));
    beta_i=zeros(length(perm(:,1)),length(perm(1,:)));
    alpha=zeros(1,length(perm(1,:)));
    Price_jcM=zeros(1,length(perm(1,:)));
    Price_jc=0;
    for jc=1:1:length(perm(1,:))
        t_i(:,jc)=tt(perm(:,jc),1);
        alpha_i(:,jc)=alphaa(perm(:,jc),1);
        beta_i(:,jc)=betaa(perm(:,jc),1);
        Vbar_i=VbarNew(perm(:,jc),1);          

        for i=1:1:nn-s+1-j 
            if i==1
                 K_i(i,jc)=Vbar_i(1,1);
            else
%               K_i(i,jc)=NCall(Vbar_i(i,1),r,vol,t_i(i,jc),t_i([i-1:-1:1],jc),K_i([i-1:-1:1],jc),ones(length(t_i([i-1:-1:1],1)),1),init,MaxStep,StepSize); 
                K_i(i,jc)=Price_s(Vbar_i(i,1),1,Vbar_i([i-1:-1:1],1),r,vol,t_i(i,jc),t_i([i-1:-1:1],jc),K_i([i-1:-1:1],jc));
            end
        end
        alpha_ii=ones(length(alpha_i(:,1)),1)-alpha_i(:,1);
        alphas=prod(alpha_ii(:,1));
        alphal=prod(alphaa([1:nn-s+1],1))/prod(alpha_i(:,1));
        alpha(1,jc)=alphas*alphal;      
                    
        Price_jcM(1,jc)=alpha(1,jc)*NCall_vol(V,r,vol*ones(length(K_i(end:-1:1,jc)),1),t0,t_i([end:-1:1],jc),K_i(end:-1:1,jc),ones(length(K_i(end:-1:1,jc)),1),zeros(length(K_i(end:-1:1,jc)),1),init,MaxStep,StepSize);     
        Price_jc=Price_jc+Price_jcM(1,jc);  
        TimeM(perm(:,jc),a)=t_i(:,jc);
        StrikeM(perm(:,jc),a)= K_i(:,jc);
        WeightM(1,a)= alpha(1,jc);
        PriceM(1,a)= Price_jcM(1,jc);       
        a=a+1;
    end   
    Price_j=Price_j+Price_jc;        
end

Price_si=Price_si+Price_j;
P_s=Price_si+resid;

            
