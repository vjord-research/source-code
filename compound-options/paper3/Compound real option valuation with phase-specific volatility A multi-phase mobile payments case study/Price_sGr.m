function [P_s,hh,aa,bb,NN,w2,w1,delta,gamma,vega,theta]=Price_sGr(V,s,Vbar,r,vol,t0,t,K,GreeksIndic)
% this function computes Ps, given s,V,Var,r,vol,t0, t vector and K vector
%V=prod(p_i(:,1))*V;
rand('seed',0);
nn=length(t);   
NN=zeros(1,nn);
aa=zeros(1,nn);aa0=-inf.*ones(1,nn);
bb=zeros(1,nn);bb0=-inf.*ones(1,nn);
Vb=zeros(1,nn);
cc_a0=-inf.*ones(1,nn);
cc_b0=-inf.*ones(1,nn);
hh=zeros(1,nn);
for j=nn:-1:s
    hh(1,nn-j+1)=t(s+nn-j,1)-t0;
    Vb(1,nn-j+1)=Vbar(s+nn-j,1);
%    bb(1,nn-j+1)=(log(V/Vbar(s+nn-j,1))+(r-vol^2/2)*hh(1,nn-j+1))/(vol*(hh(1,nn-j+1)^0.5));
%    aa(1,nn-j+1)=bb(1,nn-j+1)+vol*(hh(1,nn-j+1)^0.5);
    [aa(1,nn-j+1),bb(1,nn-j+1)]=aa_bb_f(V,Vbar(s+nn-j,1),r,vol,t(s+nn-j,1),t0);
    w2=F(s,nn-j+1,t,t0);
    NN(1,nn-j+1)=Qsimvn(nn-j+1,w2,bb0(1,1:nn-j+1),bb(1,1:nn-j+1));
end;
w1=F(s,nn-s+1,t,t0);
u=exp(-r*hh(1,1:nn-s+1)).*(K(s:nn,1))';
NK=u*(NN(1,1:nn-s+1))';
P_s=V*Qsimvn(nn-s+1,w1,aa0(1,1:nn-s+1),aa(1,1:nn-s+1))-NK;
if strcmp(GreeksIndic,'yes')==1    
    for j=nn:-1:s
        for i=nn:-1:s
            f=w1(nn-j+1,nn-i+1);
            cc_a(nn-j+1,nn-i+1)=(aa(1,nn-j+1)-f*aa(1,nn-i+1))/((1-f*f)^0.5);
            cc_b(nn-j+1,nn-i+1)=(bb(1,nn-j+1)-f*bb(1,nn-i+1))/((1-f*f)^0.5);
            
            if t(s+nn-j,1)>t(s+nn-i,1)
                [atemp,~]=aa_bb_f(V,Vb(1,nn-j+1),r,vol,t(s+nn-j,1),t(s+nn-i,1)); 
                cc_aj(nn-j+1,nn-i+1)=atemp; 
            else
                cc_aj(nn-j+1,nn-i+1)=0; 
            end;
        end;
    end;   
    delta=Qsimvn(nn-s+1,w1,aa0(1,1:nn-s+1),aa(1,1:nn-s+1));   
    gammai=zeros(1,nn-s+1);
    vegai=zeros(1,nn-s+1);
    for i=nn:-1:s  
        if i==nn
             ind=[2:1:nn-s+1]; 
             ind1=[]; 
             ind2=[2:1:nn-s+1];
             
             cc_a_bar=cc_a(ind,nn-i+1);
             cc_b_bar=cc_b(ind,nn-i+1);
        
             cc_c_bar=cc_aj(ind,nn-i+1);
        end; 
        if (i~=nn)&&(i~=s)
             ind=[[1:1:nn-i] [nn-i+2:1:nn-s+1]]; 
             ind1=[1:1:nn-i]; 
             ind2=[nn-i+2:1:nn-s+1]; 
             
             cc_a_bar=cc_a(ind,nn-i+1);
             cc_b_bar=cc_b(ind,nn-i+1);
             
             cc_c_bar=[cc_b(ind1,nn-i+1)' cc_aj(ind2,nn-i+1)'];
        end; 
        if i==s
             ind=[1:1:nn-s]; 
             ind1=[1:1:nn-s];
             ind2=[];
             
             cc_a_bar=cc_a(ind,nn-i+1);
             cc_b_bar=cc_b(ind,nn-i+1);
             
             cc_c_bar=cc_b(ind,nn-i+1);
        end; 
                
        Vbb=Vbar(s+nn-i,1);
        wd1=Dj(s,nn-s,t,t0,nn-i);
        gammai(1,nn-i+1)=(1/((2*pi)^0.5))*exp(-0.5*(aa(1,nn-i+1)^2))*(1/(V*vol*((hh(1,nn-i+1)^0.5))))*Qsimvn(nn-s,wd1,cc_a0(1,1:nn-s),cc_a_bar);      
        vegai(1,nn-i+1)=(1/((2*pi)^0.5))*exp(-0.5*(bb(1,nn-i+1)^2))*(Vbb*(hh(1,nn-i+1)^0.5))*Qsimvn(nn-s,wd1,cc_b0(1,1:nn-s),cc_c_bar);       
    end;   
    
    gamma=sum(gammai(1,:));
    
    vega=sum(vegai(1,:));
    
    theta=r*P_s-r*V*delta-0.5*vol*vol*V*V*gamma;
    
else
    if strcmp(GreeksIndic,'no')==1
        delta='na';
        gamma='na';
        vega='na';
        theta='na';
    else
        disp('Not valid greeks input indicator');       
    end;
end;





