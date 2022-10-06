% Beijing Jiaotong University
% Author: Bai
% Block fading Channel
% Channel bandwidth: 1.4MHz
% Subcarrier bandwidth: 15kHz
% Number of subcarriers: 6*12=72
% 128 FFT
% Protection of Bandwidth: 128-72=56

close all
clear all
clc
warning('off')

%====================System Parameter (constant)=========================
% Points of FFT
PO_FFT = 128;
% Cyclic Length
CYCLIC_LENGTH = 32;
% No. of TFCB
nOfTfcb = 6;
% No. of subcarriers
nOfSubCrr = nOfTfcb*12;
% No. of protective bandwidth
nOfProtctBandwidth = PO_FFT-nOfSubCrr;

%====================Optional Parameters=====================
% Modulation:1-QPSK,2-16QAM,3-64QAM
Mod_Type = 2; 
% No. of slot
NUM_slot = 7;
%Signal to noise ratio
SNR = 20;
% No. of RX antenna
NUM_RX = 1;
% No. of data symbol
nd = 18; 

% %====================User Parameter====================

% No. of resource elements 
m = 84*NUM_slot*nOfTfcb;
% No.of all users
NUM_ALL_USERS = 1000000;
% Data code-word length
md = 105;  %floor((m-mp)/nd)

%====================User Parameter====================
%Number of bits per symbol in RS coding
mRS = 3;
%Number of symbols per codeword in RS coding
nRS = 6;
%Number of symbols per message in RS coding
kRS = 4;
%RS code rate
RS_CODE_RATE = kRS / nRS;

%(2,1,6)Convolutional code rate 1/2
CONV_CODE_RATE = 1/2;
% Convolutionally encoding parameter
CONSTR_LEN = 7;% Constraint length
CODE_GEN = [171 133];% Polynomial
trellis = poly2trellis(CONSTR_LEN, CODE_GEN);

if Mod_Type==1 % QPSK symbols set
    A=[1+1j,1-1j,-1+1j,-1-1j];
    BIT_PER_SBL = 2;
elseif Mod_Type==2 % 16QAM symbols set
    A=[-3+3j,-1+3j,1+3j,3+3j,-3+1j,-1+1j,1+1j,3+1j,-3-1j,-1-1j,1-1j,3-1j,-3-3j,-1-3j,1-3j,3-3j];
    BIT_PER_SBL = 4;
elseif Mod_Type==3 % 64QAM symbols set
    A = [1+1j,1+3j,1+5j,1+7j,3+1j,3+3j,3+5j,3+7j,5+1j,5+3j,5+5j,5+7j,7+1j,7+3j,7+5j,7+7j,1-1j,1-3j,1-5j,1-7j,3-1j,3-3j,3-5j,3-7j,5-1j,5-3j,5-5j,5-7j,7-1j,7-3j,7-5j,7-7j,-1+1j,-1+3j,-1+5j,-1+7j,-3+1j,-3+3j,-3+5j,-3+7j,-5+1j,-5+3j,-5+5j,-5+7j,-7+1j,-7+3j,-7+5j,-7+7j,-1-1j,-1-3j,-1-5j,-1-7j,-3-1j,-3-3j,-3-5j,-3-7j,-5-1j,-5-3j,-5-5j,-5-7j,-7-1j,-7-3j,-7-5j,-7-7j];
    BIT_PER_SBL = 6;
end

% every users' symbols
evryUsrBts_Aftr_RS = nd*BIT_PER_SBL*CONV_CODE_RATE;

% No. of data bits = Massage bits + Group ID bits + User ID bits 
nEvryUsrDataBits = evryUsrBts_Aftr_RS * RS_CODE_RATE;
nEvryUsrIDBits= length(de2bi(NUM_ALL_USERS));  % length(de2bi(NUM_ALL_USERS))

if nEvryUsrDataBits<nEvryUsrIDBits
    error('ERROR, User sybols not enough to transmit the data and UE ID bits');
end    

% No. of symbols each user transmit in all time slots
evryUsrSbl = evryUsrBts_Aftr_RS/(BIT_PER_SBL*CONV_CODE_RATE);

% paramaters that are known in BS (for activity detectioZn and data recovery)
par.nd =nd ;
par.mRS = mRS;
par.nRS = nRS;
par.kRS = kRS;
par.trellis = trellis;
par.nEvryUsrDataBits = nEvryUsrDataBits;
par.nEvryUsrIDBits = nEvryUsrIDBits;
par.SNR = SNR;
par.NUM_RX = NUM_RX ;
par.nOfTfcb = nOfTfcb;
par.NUM_ALL_USERS = NUM_ALL_USERS;
par.Mod_Type=Mod_Type;
par.A=A;

drawPointMtxROW = 1; 
NUM_PILOT=250;

NUM_ITR = 500;

NUM_ROW  = 12 ; % the first row for num of act users
drawPointMtx1 = zeros(NUM_ROW,20);%For channel nmse
drawPointMtx2 = zeros(NUM_ROW,20);%For pilot correct ratio
drawPointMtx3 = zeros(NUM_ROW,20);%For Data correct ratio
drawPointMtx4 = zeros(NUM_ROW,20);%For pilot false alert
drawPointMtx5 = zeros(NUM_ROW,20);%For computing time

NUM_ACT_USERS=  60;
num_colunm=5;
for mp = 140 %:0.05:0.2
    num_colunm=num_colunm+1;
    
    load(['para',num2str(num_colunm),'.mat'])
    cwplt=cwplt_new(1:mp,1:NUM_PILOT)+1j*cwplt_new(mp+1:2*mp,1:NUM_PILOT);
    cwsbl=cwsbl_new(1:md,1:NUM_PILOT)+1j*cwsbl_new(md+1:2*md,1:NUM_PILOT);    
    
    for itr = 1:NUM_ITR     
        %=================Step 1: random send pilot & ID &data  =================
        % Indices of selected pilot (may be repeated)
        pilotIndx = unidrnd(NUM_PILOT,[1,NUM_ACT_USERS]);
        % Indices of selected pilot (not repeated)        
        pilot_choose =sort(unique(pilotIndx));
        % No of selected pilots             
        NUM_SELECT_PILOT=length(pilot_choose);
        acUsrIndx = sort(randperm(NUM_ALL_USERS,NUM_ACT_USERS));
        %the active user index for each pilot
        pilot_acUsr = cell(NUM_PILOT,1);
        for i = 1:NUM_ACT_USERS
             pilot_acUsr{pilotIndx(i)}= [pilot_acUsr{pilotIndx(i)};acUsrIndx(i)];
        end    
        %the pilot index for each active user
        acUsr_pilot = zeros(NUM_ALL_USERS, 1);
        acUsr_pilot(acUsrIndx,:)=pilotIndx;


        % User ID 
        usrID = zeros(NUM_ALL_USERS,nEvryUsrIDBits);
        usrID(acUsrIndx,:) = de2bi(acUsrIndx,nEvryUsrIDBits,'left-msb');
        % Generating data
        acData = zeros(NUM_ALL_USERS, nEvryUsrDataBits);
        acData(acUsrIndx,:) = randi([0,1],NUM_ACT_USERS,nEvryUsrDataBits);
        % Intigrating User ID into data
        acData(acUsrIndx,nEvryUsrDataBits-nEvryUsrIDBits+1:nEvryUsrDataBits) = usrID(acUsrIndx,:);            

        % RS Coding
        RSmatrix = reshape(acData(acUsrIndx,:)',mRS,nEvryUsrDataBits*NUM_ACT_USERS/mRS)';
        RStrSbl = bi2de(RSmatrix,'left-msb');
        hEnc = comm.RSEncoder(nRS,kRS);% RS parameter
        RS_encoded_Sbl = step(hEnc, RStrSbl);
        RS_encoded_Bits = de2bi(RS_encoded_Sbl',mRS,'left-msb');
        acdata_Aftr_RS = zeros(NUM_ALL_USERS,evryUsrBts_Aftr_RS);
        acdata_Aftr_RS(acUsrIndx,:) = reshape(RS_encoded_Bits',evryUsrBts_Aftr_RS,NUM_ACT_USERS)';

        % Convolutionally encoding
        for q = 1:NUM_ACT_USERS
            codeData(acUsrIndx(q),:) = convenc(acdata_Aftr_RS(acUsrIndx(q),:)', trellis);% Coding data
        end
        matrix = reshape(codeData(acUsrIndx,:)',6,NUM_ACT_USERS*evryUsrBts_Aftr_RS/(6*CONV_CODE_RATE));
        % Interleaving
        bits_Aftr_Itrlv = matintrlv(matrix,2,3)';

        %====================Modulation====================           
        if Mod_Type==1 % QPSK
            %输入x为0 1 2 3  输出y为1+1i 1-1i -1+1i -1-1i
            bits_Aftr_Itrlv = reshape(bits_Aftr_Itrlv',2,[])';
            dec = bi2de(bits_Aftr_Itrlv,'left-msb');
            sbl_Aftr_Mod = zeros(NUM_ALL_USERS,evryUsrSbl);
            sbl_Aftr_Mod(acUsrIndx,:) =reshape(A(dec(:,:)+1),[],NUM_ACT_USERS).';                
        elseif Mod_Type==2 % 16QAM 
            bits_Aftr_Itrlv = reshape(bits_Aftr_Itrlv',4,[])';
            dec = bi2de(bits_Aftr_Itrlv,'left-msb'); % Binary to decimal conversion
            sbl_Aftr_Mod = zeros(NUM_ALL_USERS,evryUsrSbl);
            sbl_Aftr_Mod(acUsrIndx,:) = reshape(qammod(dec,16),evryUsrSbl,NUM_ACT_USERS)';                   
        elseif Mod_Type==3 % 64QAM
            dec = bi2de(bits_Aftr_Itrlv,'left-msb'); % Binary to decimal conversion
            sbl_Aftr_Mod = zeros(NUM_ALL_USERS,evryUsrSbl);
            sbl_Aftr_Mod(acUsrIndx,:) = reshape(qammod(dec,64),evryUsrSbl,NUM_ACT_USERS)';                
        end            

        %=================Mapping to TFCB=================   
        % All users' M RXs channels in all TFCBs
        H = zeros(NUM_ALL_USERS,NUM_RX);
        %Rayleigh Channel
        H(acUsrIndx,:) = (1/sqrt(2))*(randn(NUM_ACT_USERS,NUM_RX)+1j*randn(NUM_ACT_USERS,NUM_RX));
        H = H .* Lc;

        Xsbl = repelem(sbl_Aftr_Mod,1,NUM_RX);
        Hsbl_est = mat2cell(H,[size(H,1)],NUM_RX);
        Hsbl_est = repelem(Hsbl_est,1,nd);
        Hsbl_est = cell2mat(Hsbl_est);
        HX = Hsbl_est .* Xsbl;


        % the channel for each pilot(superposed when collision occurs)
        H_pilot = zeros(NUM_PILOT,NUM_RX);
        HX_pilot = zeros(NUM_PILOT,nd*NUM_RX);
        for i = 1:NUM_ACT_USERS
            H_pilot(pilotIndx(i),:) = H_pilot(pilotIndx(i),:) + H(acUsrIndx(i),:); 
            HX_pilot(pilotIndx(i),:) = HX_pilot(pilotIndx(i),:) + HX(acUsrIndx(i),:); 
        end

        Y_pilot = cwplt*H_pilot;% Y of pilot
        Y_data = cwsbl*HX_pilot;% Y of data


        Y_pilot_rx = awgn(Y_pilot,SNR,'measured');
        Y_data_rx = awgn(Y_data,SNR,'measured');     

    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DL Receiver: LISTA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y_pilot_rx_dl=[real(Y_pilot_rx);imag(Y_pilot_rx)];
        
        [B_est_dl, H_est_dl,T_dl] = DL(Y_pilot_rx_dl,SNR,num_colunm);
        [D_est_dl] = Data_LS(B_est_dl,H_est_dl,Y_data_rx, cwsbl,NUM_PILOT,acUsrIndx,acData,par);

        nmse_dl= 10*log10(sum(abs(H_est_dl-H_pilot).^2,1)./sum(abs(H_pilot).^2,1));
        u_pilot_dl=length(intersect(B_est_dl,pilot_choose)) / length(pilot_choose);
        u_pilot_wrong_dl=(length(B_est_dl)-length(intersect(B_est_dl,pilot_choose)) )/ (length(B_est_dl)+0.0001);        
        u_data_dl=length(D_est_dl)/ NUM_ACT_USERS;
    
        drawPointMtx1(5,num_colunm) = drawPointMtx1(5,num_colunm) + nmse_dl;                   
        drawPointMtx2(5,num_colunm) = drawPointMtx2(5,num_colunm) + u_pilot_dl;                    
        drawPointMtx3(5,num_colunm) = drawPointMtx3(5,num_colunm) + u_data_dl; 
        drawPointMtx4(5,num_colunm) = drawPointMtx4(5,num_colunm) + u_pilot_wrong_dl;        
        drawPointMtx5(5,num_colunm) = drawPointMtx5(5,num_colunm) + T_dl; 
%%%%%%%%%%%%%%%%%%%%%% Dual-Net ( second iteration )%%%%%%%%%%%        
        Y_data_rx_dl=[real(Y_data_rx);imag(Y_data_rx)];
        
        [B_est_dnet, H_est_dnet,T_dnet]= DNET(Y_pilot_rx_dl,Y_data_rx_dl,SNR,num_colunm);
        [D_est_dnet] = Data_LS(B_est_dnet,H_est_dnet,Y_data_rx, cwsbl,NUM_PILOT,acUsrIndx,acData,par);

        nmse_dnet= 10*log10(sum(abs(H_est_dnet-H_pilot).^2,1)./sum(abs(H_pilot).^2,1));
        u_pilot_dnet=length(intersect(B_est_dnet,pilot_choose)) / length(pilot_choose);
        u_pilot_wrong_dnet=(length(B_est_dnet)-length(intersect(B_est_dnet,pilot_choose)) )/ length(B_est_dnet);
        u_data_dnet=length(D_est_dnet)/ NUM_ACT_USERS;
        
        drawPointMtx1(6,num_colunm) = drawPointMtx1(6,num_colunm) + nmse_dnet;                   
        drawPointMtx2(6,num_colunm) = drawPointMtx2(6,num_colunm) + u_pilot_dnet;                    
        drawPointMtx3(6,num_colunm) = drawPointMtx3(6,num_colunm) + u_data_dnet; 
        drawPointMtx4(6,num_colunm) = drawPointMtx4(6,num_colunm) + u_pilot_wrong_dnet; 
        drawPointMtx5(6,num_colunm) = drawPointMtx5(6,num_colunm) + T_dnet;   
%     %%%%%%%%%%%%%%%%%%%  AMP and ISTA Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [B_est_amp, H_est_amp,T_amp,B_est_ista, H_est_ista,T_ista] = ISTA(Y_pilot_rx_dl,cwplt_new,SNR);       

        [D_est_amp] = Data_LS(B_est_amp,H_est_amp,Y_data_rx, cwsbl,NUM_PILOT,acUsrIndx,acData,par);
        u_pilot_amp=length(intersect(B_est_amp,pilot_choose)) / length(pilot_choose);
        u_data_amp=length(D_est_amp)/ NUM_ACT_USERS;


        [D_est_ista] = Data_LS(B_est_ista,H_est_ista,Y_data_rx, cwsbl,NUM_PILOT,acUsrIndx,acData,par);
        u_pilot_ista=length(intersect(B_est_ista,pilot_choose)) / length(pilot_choose);
        u_data_ista=length(D_est_ista)/ NUM_ACT_USERS;             

        u_pilot_wrong_amp=(length(B_est_amp)-length(intersect(B_est_amp,pilot_choose))) / (length(B_est_amp)+0.0001);
        u_pilot_wrong_ista=(length(B_est_ista)-length(intersect(B_est_ista,pilot_choose))) / (length(B_est_ista)+0.0001);
        
        nmse_amp= 10*log10(sum(abs(H_est_amp-H_pilot).^2,1)./sum(abs(H_pilot).^2,1));
        nmse_ista= 10*log10(sum(abs(H_est_ista-H_pilot).^2,1)./sum(abs(H_pilot).^2,1));
        
        drawPointMtx1(1,num_colunm) = drawPointMtx1(1,num_colunm) + nmse_amp;         
        drawPointMtx1(3,num_colunm) = drawPointMtx1(3,num_colunm) + nmse_ista;         

        drawPointMtx2(1,num_colunm) = drawPointMtx2(1,num_colunm) + u_pilot_amp;         
        drawPointMtx2(3,num_colunm) = drawPointMtx2(3,num_colunm) + u_pilot_ista;         

        drawPointMtx3(1,num_colunm) = drawPointMtx3(1,num_colunm) + u_data_amp;         
        drawPointMtx3(3,num_colunm) = drawPointMtx3(3,num_colunm) + u_data_ista;    
        
        drawPointMtx4(1,num_colunm) = drawPointMtx4(1,num_colunm) + u_pilot_wrong_amp;         
        drawPointMtx4(3,num_colunm) = drawPointMtx4(3,num_colunm) + u_pilot_wrong_ista;         

        drawPointMtx5(1,num_colunm) = drawPointMtx5(1,num_colunm) + T_amp;         
        drawPointMtx5(3,num_colunm) = drawPointMtx5(3,num_colunm) + T_ista;         
        
    end
end

drawPointMtx1  = drawPointMtx1 ./ NUM_ITR;
drawPointMtx2  = drawPointMtx2 ./ NUM_ITR;
drawPointMtx3  = drawPointMtx3 ./ NUM_ITR;
drawPointMtx4  = drawPointMtx4 ./ NUM_ITR;
drawPointMtx5  = drawPointMtx5 ./ NUM_ITR;
save('time_1','drawPointMtx5');
