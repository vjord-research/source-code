function [P,Vbar]=NDigitalCNPlot(VGrid,r,vol_i,t00,t_i,K_i,p_i,d_i,init,MaxStep,StepSize,GraphsIndic)
rand('seed',0);
n=length(t_i);
N=length(VGrid);
P=zeros(n,N);

for s=n:-1:1
    for i=1:1:N
         [P(s,i),Vbar]=NDigitalCN_vol(VGrid(i),r,vol_i([s:end],1),t00,t_i([s:end],1),K_i([s:end],1),p_i([s:end],1),d_i([s:end],1),init,MaxStep,StepSize);    
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
        plot(VGrid,P(s,:),'Color',lns{1,s},'LineStyle','-','LineWidth',emph1)
    end
    hold on
end
xlabel('V')
ylabel('Price: C(V,t)')
title('Compound Cash or Nothing Digital')
j=pp';
leg2=leg(fliplr(j));
legend(leg2,'FontSize',8)




