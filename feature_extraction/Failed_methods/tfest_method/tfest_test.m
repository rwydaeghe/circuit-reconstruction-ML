t=linspace(0,30,101);
x=exp(-t);
data=iddata(x.',t.');
%plot(data);
sys = tfest(data, 1, 0);
%figure;
%impulseplot(sys);

load iddata1 z1;
np = 2;
%plot(z1)
sys = tfest(z1,np)
lsim(sys,z1);
