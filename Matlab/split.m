for i=740:1219,
I=imread(['OUT' num2str(i) ,'.png']);
I1=I(1:480,1:640,:);
imwrite(I1, ['OL' num2str(i),'.png']);
imwrite(I(1:480,641:1280,:), ['OR' num2str(i),'.png']);
end