def afagps(x,s,name):
    z=ifft(fft(x)*fft(x[::-1]))
    r=list()
    for i in range(int(len(x)/2)):
        r.append(z[i]/(len(x)/2-i))
    rk=r/r[0]
    for i in range(len(rk)):
        if abs(rk[i])<=s:
            rk[i]=0
    ark=np.array(rk)
    rr=np.hstack((ark[::-1],np.delete(ark,[0])))
    a=fft(rr)
    p=list()
    for i in range(len(a)):
        p.append(a[i]*math.e**(1j*i*math.pi*len(x)/2/(2*len(x/2)+1)))
    pp=list()
    for i in range(int(len(x)/4)):
        pp.append(p[2*i])
    plt.figure(figsize=(20,10))
    plt.savefig("C:\\Users\\Lenovo\\Desktop\\气象水文\\"+name+".png")
    plt.plot(pp)