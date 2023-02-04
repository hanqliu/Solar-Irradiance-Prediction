file="D:\e\南开光伏课题数据集及说明\NK2_GF\训练数据\气象数据\Station_1.csv"
%x=csvread(file,1,2,[1,2,10000,2])
Fs=100;%采样率
t = 0:1/Fs:10-1/Fs;
x = sin(2*pi*10*t)
k=300
z=ifft(fft(x).*fft(x(end:-1:1)))
r=[];
for m = 1:k
    m=x(m)/(2*k-m);
    r=[r m];
end
r=r/max(r);
for m =1:length(r)
    if(r(m)<=0.3)
        r(m)=0;
    end
end
a=[];
rx=r;
rx(1)=[];
a=[r(end:-1:1) rx];
fa=fft(a);
p=[];
for m =1:length(fa)
    p=[p fa(m)*exp(1i*m*pi*k/(2*k+1))];
end
re=[];
for m=1:length(fa)/2
     re=[re p(2*m-1)];
end
plot(abs(re))