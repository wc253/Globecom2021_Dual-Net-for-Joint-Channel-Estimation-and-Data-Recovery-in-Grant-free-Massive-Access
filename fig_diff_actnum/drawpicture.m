% draw picture 1
load('result.mat')

figure(1)
plot(20:20:140,drawPointMtx1(3,1:7),'-o',...
    20:20:140,drawPointMtx1(1,1:7),'-+',...
    20:20:140,drawPointMtx1(5,1:7),'-s',...
    20:20:140,drawPointMtx1(6,1:7),'-v',...
   'LineWidth',2,'MarkerSize',8);
legend('ISTA','AMP','LISTA','Dual-Net');
xlabel('Number of active users')
ylabel('NMSE (dB)')
grid on

figure(2)
plot(20:20:140,drawPointMtx3(3,1:7),'-o',...
    20:20:140,drawPointMtx3(1,1:7),'-+',...
    20:20:140,drawPointMtx3(5,1:7),'-s',...
    20:20:140,drawPointMtx3(6,1:7),'-v',...
   'LineWidth',2,'MarkerSize',8);
legend('ISTA','AMP','LISTA','Dual-Net');
xlabel('Number of active users')
ylabel('\mu_{data}(%)')
grid on
