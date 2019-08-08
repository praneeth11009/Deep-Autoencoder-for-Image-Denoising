yp2=ypred(:,:,:,1:8);
yp2=reshape(yp2,[32,32,8]);
yc=num2cell(yp2,[1,2]);
yc=reshape(yc,[8,1]);
minibatch1 = cat(3,yc,inputImageExamples(:,1),inputImageExamples(:,2));
figure;
montage(minibatch1,'Size',[3 8])