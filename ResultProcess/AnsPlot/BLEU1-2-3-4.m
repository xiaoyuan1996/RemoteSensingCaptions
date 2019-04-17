clear
clc

%% Definition
LINE_WIDTH = 2;
MARKER_SIZE = 5;
END_PLACE = 50+1;

%%


A=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\Resnet_Sydney\Save_Excel.xlsx');
B=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\Resnet_UCM\Save_Excel.xlsx');
C=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\VGG_Sydney\Save_Excel.xlsx');
D=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\VGG_UCM\Save_Excel.xlsx');
 X=0:END_PLACE-1;
 subplot(2,2,1);
 xlabel('Epoch')
 %plot(X,A(1:25,4),'-ko','LineWidth',2);
 %set(gca,'linewidth',1.5);
 %hold on
 plot(X,A(1:END_PLACE,5),'-r^','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,A(1:END_PLACE,6),'-g+','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','g');
  hold on
 plot(X,A(1:END_PLACE,7),'-ms','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,A(1:END_PLACE,8),'-bp','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','b');
 axis([0 END_PLACE,0,1])
 title('Resnet-Sydney','FontSize',20)
  grid on;
 hold off
  xlabel('Epoch')
 subplot(2,2,2);
 

 plot(X,B(1:END_PLACE,5),'-r^','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,B(1:END_PLACE,6),'-g+','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','g');
  hold on
 plot(X,B(1:END_PLACE,7),'-ms','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,B(1:END_PLACE,8),'-bp','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','b');
 axis([0 END_PLACE,0,1])
 title('Resnet-UCM','FontSize',20)
  grid on;
 hold off
  xlabel('Epoch')
  subplot(2,2,3);

 plot(X,C(1:END_PLACE,5),'-r^','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,C(1:END_PLACE,6),'-g+','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','g');
  hold on
 plot(X,C(1:END_PLACE,7),'-ms','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,C(1:END_PLACE,8),'-bp','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','b');
 axis([0 END_PLACE,0,1])
 title('VGG-Sydney','FontSize',20)
  grid on;
 hold off
  xlabel('Epoch')
  subplot(2,2,4);

 plot(X,D(1:END_PLACE,5),'-r^','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,D(1:END_PLACE,6),'-g+','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','g');
  hold on
 plot(X,D(1:END_PLACE,7),'-ms','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,D(1:END_PLACE,8),'-bp','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','b');
 axis([0 END_PLACE,0,1])
  xlabel('Epoch')
  title('VGG-UCM','FontSize',20)
  hold off 
 hleg2 = legend('bleu4','bleu3','bleu2','bleu1','northwest','NorthEastOutside')
 set(hleg2,'Fontsize',15);
 set(hleg2, 'Position', [0.489620536745393 0.471757337519333 0.0421874995343388 0.0722453202626313]);
legend('boxoff')
 grid on;


