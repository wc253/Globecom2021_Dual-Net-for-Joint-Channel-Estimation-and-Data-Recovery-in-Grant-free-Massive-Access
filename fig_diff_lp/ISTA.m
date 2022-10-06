function [B_est_amp, H_est_amp,T_amp,B_est_ista, H_est_ista,T_ista]  = ISTA(y, A,SNR)

L = 1;
[M,N] = size(A);
mp=M/2;
NUM_PILOT=N/2;
% algorithm parameters
T = 20; % AMP iterations
Tii =1000; % ISTA iterations
alf = 1.1402; % amp tuning parameter [1.1402]
eta = @(r,lam) sign(r).*max(bsxfun(@minus,abs(r),lam),0); 

erre=10^(-SNR/10);

tic
% run AMP
Bmf = A'; % matched filter 
xhat = zeros(N,L); % initialization of signal estimate
v = zeros(M,L); % initialization of residual
for t=1:T
    g = (N/M)*mean(xhat~=0,1); % onsager gain
    v = y - A*xhat + bsxfun(@times,v,g); % residual
    rhat = xhat + Bmf*v; % denoiser input
    rvar = sum(abs(v).^2,1)/M; % denoiser input err var
    xhat = eta(rhat, alf*sqrt(rvar)); % estimate
end

H_amp=xhat(1:N/2)+1i*xhat(N/2+1:N);
[H_sort,pos]=sort(sum(abs(H_amp).^2,2),'descend');
B_est_amp=pos(H_sort>erre);

H_est_amp=zeros(N/2,1);
H_est_amp(B_est_amp,:)=H_amp(B_est_amp,:);
toc
T_amp=toc;

tic
lam_mf = 0.1;
% run ISTA
scale = .999/norm(Bmf*A);
B = scale*Bmf;
xhat = zeros(N,L); % initialization of signal estimate
for t=1:Tii
    v = y - A*xhat; % residual
    rhat = xhat + B*v; % denoiser input
    xhat = eta(rhat, lam_mf*scale); % estimate
end

H_ista=xhat(1:N/2)+1i*xhat(N/2+1:N);
[H_sort,pos]=sort(sum(abs(H_ista).^2,2),'descend');
B_est_ista=pos(H_sort>erre);

H_est_ista=zeros(N/2,1);
H_est_ista(B_est_ista,:)=H_ista(B_est_ista,:);
toc
T_ista=toc;


