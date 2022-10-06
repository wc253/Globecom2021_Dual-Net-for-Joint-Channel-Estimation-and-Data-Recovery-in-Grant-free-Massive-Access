function [B_est, H_est,T_dnet] = DNET(Y,Yd,SNR,i)

load(['dnetpara',num2str(i),'.mat']);
% load(['netpara.mat']);

[M,N] = size(B_0);

erre=10^(-SNR/10);

tic
By=B_0*Y;
By1=B1_0'*Yd;
xcombin=[By,By1].*max(1-lam_0./sqrt(sum(abs([By,By1]).^2,2)),0);
xhat=xcombin(:,1);
xhat1=xcombin(:,2:end);
for t=1:19
    lam=eval(['lam_',num2str(t)]);
    xcombin=[S_0*xhat+By,S1_0'*xhat1+By1].*max(1-lam./sqrt(sum(abs([S_0*xhat+By,S1_0'*xhat1+By1]).^2,2)),0);
    xhat=xcombin(:,1);
    xhat1=xcombin(:,2:end);  
end

H=xhat(1:M/2)+1i*xhat(M/2+1:M);
[H_sort,pos]=sort(sum(abs(H).^2,2),'descend');
B_est =pos(H_sort>erre);
H_est=zeros(M/2,1);
H_est(B_est,:)=H(B_est,:);
toc
T_dnet=toc;


