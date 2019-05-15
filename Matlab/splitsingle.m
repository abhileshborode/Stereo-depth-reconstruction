I=imread('a29.png');
I1=I(1:480,1:640,:);
imwrite(I1, ['aaaL29.png']);
imwrite(I(1:480,641:1280,:),['aaaR29.png']);