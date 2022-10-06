function [B_est, H_est,T_dl] = DL(Y,SNR,i)

eta = @(r,lam) sign(r).*max(bsxfun(@minus,abs(r),lam),0); 
load(['listapara',num2str(i),'.mat']);
% load(['netpara.mat']);

[M,N] = size(B_0);
erre=10^(-SNR/10);

tic
By=B_0*Y;
xhat=eta(By,lam_0);

for t=1:19
    lam=eval(['lam_',num2str(t)]);
    xhat=eta( S_0*xhat+By,lam);
end

H=xhat(1:M/2)+1i*xhat(M/2+1:M);
[H_sort,pos]=sort(sum(abs(H).^2,2),'descend');
B_est =pos(H_sort>erre);
H_est=zeros(M/2,1);
H_est(B_est,:)=H(B_est,:);
toc
T_dl=toc;




