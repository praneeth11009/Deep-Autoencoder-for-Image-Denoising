function img = poison_noise(img)
N=16;
img = poissrnd(im2double(img)*N); 
img = img/max(img(:));
end
