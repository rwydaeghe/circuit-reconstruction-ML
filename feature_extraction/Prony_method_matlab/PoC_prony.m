close all
warning('off','all')
global x, global t
%zeros=[-0.5,-1.5,-2.5,-3.5,-4.5,-5.5,-6.5,-7.5];
%poles=[-1+1i,-1-1i,-2,3i,-3i,-4,-5+5i,-5-5i];
N=16;
zeros=linspace(-1.5,-N-0.5,N).';
poles=linspace(-1,-N,2*N-1).';
poles(1:2:end);
poles(1:2:end)=poles(1:2:end)+1i*poles(1:2:end);
poles(2:2:end-1);
poles(2:2:end-1)=floor(poles(2:2:end-1))-1i*floor(poles(2:2:end-1));
poles(1)=-1;
% usually it fails for high (>8) amounts of poles ánd the amount of zeros
% is about about equal to number of poles
nq_sel=3;
np_sel=7; %needs to be uneven
zeros=zeros(1:nq_sel);
poles=poles(1:np_sel);
%zeros=[-0.5,+1.5];
%poles=[-1+1i,-1-1i,-2];
gain=3.5;
sys=zpk(zeros,poles,gain);
Tfinal=10;
dt=1/200;
t=linspace(0,Tfinal,Tfinal/dt+1).';
x=impulse(sys,t);
plot(t,x)

max_np=2*length(poles); %arbitrary. seems to give best results.
wb = waitbar(0,'Please wait...');
err_best=inf;
err_plot=[]; nq_plot=[]; np_plot=[];
figure
hold on;
plot(t,x)
ylim([min(x)*1.5, max(x)*1.5])
y_old=x;
for np=1:max_np
    waitbar(np/max_np,wb,"Sweeping all zeros for "+num2str(np)+" poles...");
    for nq=1:np
        [Ha, y]=continuous_prony(x,nq,np);
        err=mean((x-y).^2);
        if err<err_best
            err_best=err;
            Ha_best=Ha;
            err_plot(end+1)=err;
            nq_plot(end+1)=nq;
            np_plot(end+1)=np;
            y_best=y;
            plot(t,y_best)
        end
    end
end
waitbar(1,wb,'Post-processing')

% error evolution plot along with nq, np
figure
loglog(1:length(err_plot), err_plot)
for i=1:length(err_plot)
    text(i, err_plot(i), "("+nq_plot(i)+","+np_plot(i)+")")
end

% plot the final fit and poles and zeros
figure
hold on
plot(t,x)
plot(t,y_best)
predicted_gain=Ha_best.K;
predicted_zeros=zero(Ha_best);
predicted_poles=pole(Ha_best);
gain
zeros
poles
predicted_gain
predicted_zeros
predicted_poles
nrse_in_gain=abs(predicted_gain-gain)/abs(gain);
if length(predicted_zeros)~=length(zeros)
    disp('The number of zeros predicted is wrong')
    min_len=min(length(zeros),length(predicted_zeros));
    nrmse_in_zeros=sqrt(mean(abs(set_imag_part_pos(predicted_zeros(1:min_len))-set_imag_part_pos(zeros(1:min_len))).^2))/my_range(abs(set_imag_part_pos(zeros(1:min_len))));
else
    nrmse_in_zeros=sqrt(mean(abs(set_imag_part_pos(predicted_zeros)-set_imag_part_pos(zeros)).^2))/my_range(abs(set_imag_part_pos(zeros)));
end
if length(predicted_poles)~=length(poles)
    disp('The number of poles predicted is wrong')
    min_len=min(length(poles),length(predicted_poles));
    nrmse_in_poles=sqrt(mean(abs(set_imag_part_pos(predicted_poles(1:min_len))-set_imag_part_pos(poles(1:min_len))).^2))/my_range(abs(set_imag_part_pos(poles(1:min_len))));
else
    nrmse_in_poles=sqrt(mean(abs(set_imag_part_pos(predicted_poles)-set_imag_part_pos(poles)).^2))/my_range(abs(set_imag_part_pos(poles)));
end
nrse_in_gain
nrmse_in_zeros
nrmse_in_poles
percent_wrong_deemed_acceptabe=10;
if nrmse_in_zeros>percent_wrong_deemed_acceptabe/100
    disp('I probably got the values of zeros wrong')
end
if nrmse_in_poles>percent_wrong_deemed_acceptabe/100
    disp('I probably got the values of poles wrong')
end
if nrse_in_gain>percent_wrong_deemed_acceptabe/100
    disp('I probably got the value of gain wrong')
end
close(wb)

function out=set_imag_part_pos(in)
out=[];
for i=1:length(in)
    if imag(in(i))<0
        in(i)=real(in(i))-1i*imag(in(i));
    end
    out(end+1)=in(i);
end
end

function out=my_range(x)
if size(x)==1
    out=x;
else
    out=max(x)-min(x);
end
end

function [Ha, y] = continuous_prony(x,nq,np)
global x, global t
dt=t(2)-t(1);
[b,a]=prony(x, nq, np); 
Hd=tf(b,a,dt);
%Use tustin transform. Reduces precision of location so needs enough sampling. zoh/foh has its problems (see literature)
Ha=d2c(Hd,'tustin'); 
zeros=flip(zero(Ha));
poles=flip(pole(Ha));
cut_off_sampling_zp=true;
if cut_off_sampling_zp
    z_i_cut_off=length(zeros);
    niquist_freq=1/(2*dt);
    for i=1:length(zeros)
        % theoretically I'd do abs(imag(z)) but just abs works better. May cut
        % some large zeros or poles off that shouldn't be. but if you just sample
        % enough than this shouldn't ever be an issue
        if abs(zeros(i))>niquist_freq 
            z_i_cut_off=i-1;
            break
        end
    end
    p_i_cut_off=length(poles);
    for i=1:length(poles)
        if abs(poles(i))>niquist_freq
            p_i_cut_off=i-1;
            break
        end
    end
    zeros=zeros(1:z_i_cut_off);
    poles=poles(1:p_i_cut_off);
end
Ha=zpk(zeros,poles,1);
y=impulse(Ha,t);
if ~cut_off_sampling_zp
    y=y*dt;
end    
gain=x(ceil(end/2))/y(ceil(end/2)); % this works due to linearity of laplace tf. hope that y point isnt zero (to do)
Ha=zpk(zeros,poles,gain);
y=y*gain; %ditto, no double calculations necessary :)
end    