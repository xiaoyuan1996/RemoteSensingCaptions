clear
clc

%% Definition
LINE_WIDTH = 2;
MARKER_SIZE = 5;
END_PLACE = 25+1;

%%

C=xlsread('D:\Python\tensorflow\image caption coding\ResultProcess\VGG_Sydney\Save_Excel.xlsx');

 X=0:END_PLACE-1;



 plot(X,C(1:END_PLACE,5),'-r^','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','r');
 hold on
 plot(X,C(1:END_PLACE,6),'-g+','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','g');
  hold on
 plot(X,C(1:END_PLACE,7),'-ms','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','m');
 hold on
 plot(X,C(1:END_PLACE,8),'-bp','LineWidth',LINE_WIDTH,'MarkerSize',	MARKER_SIZE,'MarkerFaceColor','w','MarkerEdgeColor','b');
 axis([0 END_PLACE,0,1])
   plot(X,C(1:END_PLACE,4),':ks','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','k');
   hold on
    plot(X,C(1:END_PLACE,3),'--ks','LineWidth',2,'MarkerSize',	6,'MarkerFaceColor','w','MarkerEdgeColor','k');
 hold on

 title('VGG-Sydney','FontSize',20)
  grid on;
 hold off
  axis([0,END_PLACE,0,2.2]);
 
  hleg2 = legend('bleu4','bleu3','bleu2','bleu1','ROUGE-L','CIDEr','northwest','NorthEastOutside')
 set(hleg2,'Fontsize',15);
 set(hleg2, 'Position', [0.145870537226576 0.723836341023394 0.0744791652386387 0.16891891408611]);
legend('boxoff')
set(gca,'XMinorGrid','on')
set(gca,'YMinorGrid','on')
xlabel('Epoch')


