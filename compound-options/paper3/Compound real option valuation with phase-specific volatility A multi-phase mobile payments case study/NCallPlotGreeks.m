function [P]=NCallPlotGreeks(VGrid,r,vol_i,t00,t_i,K_i,p_i,d_i,init,MaxStep,StepSize,GreeksIndic,GraphsIndic)
rand('seed',0);
n=length(t_i);
N=length(VGrid);
P=zeros(n,N);
delta=zeros(n,N);
gamma=zeros(n,N);
vega=zeros(n,N);
theta=zeros(n,N);
d_ii=1-d_i;

for s=n:-1:1
    if s==n
        for i=1:1:N
            Vi=prod(p_i(:,1))*VGrid(i)*prod(d_ii(:,1));
            P(s,i)=blsprice(Vi, K_i(end), r, t_i(end)-t00,vol_i);
            delta(s,i)=blsdelta(Vi,K_i(end), r, t_i(end)-t00,vol_i);
            gamma(s,i)=blsgamma(Vi,K_i(end), r, t_i(end)-t00,vol_i);
            vega(s,i)=blsvega(Vi,K_i(end), r, t_i(end)-t00,vol_i); 
            theta(s,i)=blstheta(Vi,K_i(end), r, t_i(end)-t00,vol_i);
        end       
    else
        for i=1:1:N
            [P(s,i),~,delta(s,i),gamma(s,i),vega(s,i),theta(s,i)]=NCall_temp(VGrid(i),r,vol_i,t00,t_i([s:end],1),K_i([s:end],1),p_i([s:end],1),d_i([s:end],1),init,MaxStep,StepSize,GreeksIndic);
                                                                              
        end
    end
end

lns={'green','red','blue','magenta','cyan','yellow','black','green','red','blue','magenta','cyan'};
cc={':','--','-'};
leg={'C(1); n-fold','C(2)','C(3)','C(4)','C(5)','C(6)','C(7)','C(8)','C(9)','C(10)','C(11)','C(12)','C(13)','C(14)','C(15)','C(16)','C(17)','C(18)','C(19)','C(20)'};
emph1=1.5;

pp=find(GraphsIndic);

figure(1)

for s=n:-1:1
    if GraphsIndic(s,1)==1
        plot(VGrid,delta(s,:),'Color',lns{1,s},'LineStyle','-','LineWidth',emph1)
    end;
    hold on
end;
xlabel('V')
ylabel('Delta')
title('Delta')

j=pp';

leg2=leg(fliplr(j));
legend(leg2,'FontSize',8)

figure(2)
for s=n:-1:1
    if GraphsIndic(s,1)==1
        plot(VGrid,gamma(s,:),'Color',lns{1,s},'LineStyle','-','LineWidth',emph1)
    end
    hold on
end
xlabel('V')
ylabel('Gamma')
title('Gamma')
j=pp';

leg2=leg(fliplr(j));
legend(leg2,'FontSize',8)

figure(3)
for s=n:-1:1
    if GraphsIndic(s,1)==1
        plot(VGrid,vega(s,:),'Color',lns{1,s},'LineStyle','-','LineWidth',emph1)
    end
    hold on
end
xlabel('V')
ylabel('Vega')
title('Vega')
j=pp';

leg2=leg(fliplr(j));
legend(leg2,'FontSize',8)

figure(4)
for s=n:-1:1
    if GraphsIndic(s,1)==1
        plot(VGrid,theta(s,:),'Color',lns{1,s},'LineStyle','-','LineWidth',emph1)
    end
    hold on
end
xlabel('V')
ylabel('Theta')
title('Theta')
j=pp';

leg2=leg(fliplr(j));
legend(leg2,'FontSize',8)


