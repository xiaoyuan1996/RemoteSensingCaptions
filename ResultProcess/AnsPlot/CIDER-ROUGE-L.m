clear
clc

%% Defination
END_PLACE = 50+1;

%%

A=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\Resnet_Sydney\Save_Excel.xlsx');
B=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\Resnet_UCM\Save_Excel.xlsx');
C=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\VGG_Sydney\Save_Excel.xlsx');
D=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\VGG_UCM\Save_Excel.xlsx');
 X=0:END_PLACE-1;
 plot(X,A(1:END_PLACE,3),'-r^','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,B(1:END_PLACE,3),'-g+','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','g');
 hold on
 plot(X,C(1:END_PLACE,3),'-ms','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,D(1:END_PLACE,3),'-bp','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','b');
 hold on
 grid on
hleg2 = legend('Resnet-Sydney','Resnet-UCM','VGG-Sydney','VGG-UCM','Location','SouthEast')
 set(hleg2,'Fontsize',20);
    xlabel('Epoch')

 title('CIDEr','FontSize',20)
 figure(2)
 plot(X,A(1:END_PLACE,4),'-r^','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
  plot(X,B(1:END_PLACE,4),'-g+','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','g');
 hold on
  plot(X,C(1:END_PLACE,4),'-ms','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
  plot(X,D(1:END_PLACE,4),'-bp','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','b');
  grid on
 hleg2 =legend('Resnet-Sydney','Resnet-UCM','VGG-Sydney','VGG-UCM','Location','SouthEast')
  set(hleg2,'Fontsize',20);
  title('ROUGE-L','FontSize',20)
      xlabel('Epoch')