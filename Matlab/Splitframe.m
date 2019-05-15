clc;
clear all;
close all;
tic;
vid=VideoReader('OUT8.avi');
  numFrames = vid.NumberOfFrames;
  n=numFrames;
  j=1086;
for i = 1:50:n
  frames = read(vid,i);
   imwrite(frames,['OUT' int2str(j), '.png']);
   j=j+1;
end 
