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
NUM_slot = 4;
%Signal to noise ratio
SNR = 20;
% No. of RX antenna
NUM_RX = 1;
% Length of Pilot in each RB*NUM_slot
mp =120;
% No. of data symbol
nd = 18; 
% % transmit power(dbm) 
% rho=23;
% 
% %====================User Parameter====================
% %total transmit energy
% xi=sqrt((mp*rho)); 

% No. of resource elements 
m = 84*NUM_slot*nOfTfcb;
% No.of all users
NUM_ALL_USERS = 1000000;
% Data code-word length
md = 105;  %floor((m-mp)/nd)
% No. of empty RB elements
eptymd = m-mp-nd*md; 

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

load('para.mat')
cwplt=cwplt_new(1:mp,1:NUM_PILOT)+1j*cwplt_new(mp+1:2*mp,1:NUM_PILOT);
cwsbl=cwsbl_new(1:md,1:NUM_PILOT)+1j*cwsbl_new(md+1:2*md,1:NUM_PILOT);

NUM_BATCH=500;
NUM_ITR =2000; % batchsize
for NUM_ACT_USERS= 20 %:20:140
    for bat=1:NUM_BATCH  %:0.05:0.2
        y=zeros(2*mp,NUM_ITR);
        x=zeros(2*NUM_PILOT,NUM_ITR);
%         y1=zeros(NUM_ITR,nd,2*md);
%         x1=zeros(NUM_ITR,nd,2*NUM_PILOT);
        for itr = 1:NUM_ITR
            %=================Step 1: random send pilot & ID &data  =================
            % Indices of selected pilot (may be repeated)
            pilotIndx = unidrnd(NUM_PILOT,[1,NUM_ACT_USERS]);
            % Indices of selected pilot (not repeated)        
            pilot_choose =sort(unique(pilotIndx));
            % No of selected pilots             
            NUM_SELECT_PILOT=length(pilot_choose);
            % Indices of active users
            acUsrIndx = sort(randperm(NUM_ALL_USERS,NUM_ACT_USERS));

            % User ID 
            usrID = zeros(NUM_ACT_USERS,nEvryUsrIDBits);
            usrID = de2bi(acUsrIndx,nEvryUsrIDBits,'left-msb');
            % Generating data
            acData = zeros(NUM_ACT_USERS, nEvryUsrDataBits);
            acData = randi([0,1],NUM_ACT_USERS,nEvryUsrDataBits);
            % Intigrating User ID into data
            acData(:,nEvryUsrDataBits-nEvryUsrIDBits+1:nEvryUsrDataBits) = usrID;            

            % RS Coding
            RSmatrix = reshape(acData',mRS,nEvryUsrDataBits*NUM_ACT_USERS/mRS)';
            RStrSbl = bi2de(RSmatrix,'left-msb');
            hEnc = comm.RSEncoder(nRS,kRS);% RS parameter
            RS_encoded_Sbl = step(hEnc, RStrSbl);
            RS_encoded_Bits = de2bi(RS_encoded_Sbl',mRS,'left-msb');
            acdata_Aftr_RS = zeros(NUM_ACT_USERS,evryUsrBts_Aftr_RS);
            acdata_Aftr_RS = reshape(RS_encoded_Bits',evryUsrBts_Aftr_RS,NUM_ACT_USERS)';

            % Convolutionally encoding
            for q = 1:NUM_ACT_USERS
                codeData(acUsrIndx(q),:) = convenc(acdata_Aftr_RS(q,:)', trellis);% Coding data
            end
            matrix = reshape(codeData(acUsrIndx,:)',6,NUM_ACT_USERS*evryUsrBts_Aftr_RS/(6*CONV_CODE_RATE));
            % Interleaving
            bits_Aftr_Itrlv = matintrlv(matrix,2,3)';

            %====================Modulation====================           
            if Mod_Type==1 % QPSK
                %输入x为0 1 2 3  输出y为1+1i 1-1i -1+1i -1-1i
                bits_Aftr_Itrlv = reshape(bits_Aftr_Itrlv',2,[])';
                dec = bi2de(bits_Aftr_Itrlv,'left-msb');
                sbl_Aftr_Mod = zeros(NUM_ACT_USERS,evryUsrSbl);
                sbl_Aftr_Mod =reshape(A(dec(:,:)+1),[],NUM_ACT_USERS).';                
            elseif Mod_Type==2 % 16QAM 
                bits_Aftr_Itrlv = reshape(bits_Aftr_Itrlv',4,[])';
                dec = bi2de(bits_Aftr_Itrlv,'left-msb'); % Binary to decimal conversion
                sbl_Aftr_Mod = zeros(NUM_ACT_USERS,evryUsrSbl);
                sbl_Aftr_Mod= reshape(qammod(dec,16),evryUsrSbl,NUM_ACT_USERS)';                   
            elseif Mod_Type==3 % 64QAM
                dec = bi2de(bits_Aftr_Itrlv,'left-msb'); % Binary to decimal conversion
                sbl_Aftr_Mod = zeros(NUM_ACT_USERS,evryUsrSbl);
                sbl_Aftr_Mod = reshape(qammod(dec,64),evryUsrSbl,NUM_ACT_USERS)';                
            end            

            %=================Mapping to TFCB=================   
            % All users' M RXs channels in all TFCBs
            H = zeros(NUM_ACT_USERS,NUM_RX);
            %Rayleigh Channel
            H = (1/sqrt(2))*(randn(NUM_ACT_USERS,NUM_RX)+1j*randn(NUM_ACT_USERS,NUM_RX));
            H = H .* Lc(acUsrIndx,:);

            Xsbl = repelem(sbl_Aftr_Mod,1,NUM_RX);
            Hsbl_est = mat2cell(H,[size(H,1)],NUM_RX);
            Hsbl_est = repelem(Hsbl_est,1,nd);
            Hsbl_est = cell2mat(Hsbl_est);
            HX = Hsbl_est .* Xsbl;

            % the channel for each pilot(superposed when collision occurs)
            H_pilot = zeros(NUM_PILOT,NUM_RX);
            HX_pilot = zeros(NUM_PILOT,nd*NUM_RX);
            for i = 1:NUM_ACT_USERS
                H_pilot(pilotIndx(i),:) = H_pilot(pilotIndx(i),:) + H(i,:); 
                HX_pilot(pilotIndx(i),:) = HX_pilot(pilotIndx(i),:) + HX(i,:); 
            end

            Y_pilot = cwplt*H_pilot;% Y of pilot
            Y_data = cwsbl*HX_pilot;% Y of data

            Y_pilot_rx = awgn(Y_pilot,SNR,'measured');
            Y_data_rx = awgn(Y_data,SNR,'measured');     

            y(:,itr)=[real(Y_pilot_rx);imag(Y_pilot_rx)];
            x(:,itr)=[real(H_pilot);imag(H_pilot)];
%             y1(itr,:,:)=[real(Y_data_rx);imag(Y_data_rx)]';
%             x1(itr,:,:)=[real(HX_pilot);imag(HX_pilot)]';     
        end
    save(['data/actnum',num2str(NUM_ACT_USERS),'/data',num2str(bat),'.mat'],'y','x');        
%     save(['data/actnum',num2str(NUM_ACT_USERS),'/data',num2str(bat),'.mat'],'y','x','y1','x1');
        
    end
end
