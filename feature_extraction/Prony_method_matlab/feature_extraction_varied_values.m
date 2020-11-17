close all
warning('off','all')
wb = waitbar(0,'Please wait...');
global t
load varied_values_data_set
t=t;
N=length(data(1,:));
all_zeros=zeros(1,N);
all_poles=zeros(2,N);
all_gains=zeros(1,N);
plotting_on=false; %you definitely don't want this for big data sets
for i=1:N
    x_i=data(:,i);
    max_np=2*2; %2x the expected amount. arbitrary. seems to give best results.
    err_best=inf;
    err_plot=[]; nq_plot=[]; np_plot=[];
    waitbar(i/N,wb,"Computing zeros/poles/gain of transient "+num2str(i)+"/"+num2str(N)+"...");
    for np=1:max_np
        for nq=1:np
            [Ha, y]=continuous_prony(x_i,nq,np);
            err=mean((x_i-y).^2);
            if err<err_best
                err_best=err;
                Ha_best=Ha;
                err_plot(end+1)=err;
                nq_plot(end+1)=nq;
                np_plot(end+1)=np;
                y_best=y;
            end
        end
    end
    if plotting_on
        figure(2*i-1)
        hold on;
        ylim([min(x_i)*1.5, max(x_i)*1.5])
        plot(t,x_i)
        plot(t,y_best)
    end
    % error evolution plot along with nq, np
    if plotting_on
        figure(2*i)
        hold on;
        loglog(1:length(err_plot), err_plot)
        for i=1:length(err_plot)
            text(i, err_plot(i), "("+nq_plot(i)+","+np_plot(i)+")")
        end
    end
    foo=size(zero(Ha_best));
    if foo(1)== 1
        all_zeros(:,i)=zero(Ha_best);
    end
    foo=size(pole(Ha_best));
    if foo(1)== 2
        all_poles(:,i)=pole(Ha_best);
    end
    all_gains(:,i)=Ha_best.K;
end
close(wb)
all_zeros
all_poles
all_gains
all_pre=1/2*(all_poles+conj(all_poles));
all_pim=1i/2*((-all_poles+conj(all_poles))); 
% I don't have a good way of doing the following in general, but it's
% probaly not necessary
% basically for this topology I know there's only 2 poles, one gain and one
% zero (that's always equal to zero)
all_pre=all_pre(1,:);
all_pim=all_pim(2,:); %positive imag part!!
features=zeros(3,N); 
features(1,:)=all_pre;
features(2,:)=all_pim;
features(3,:)=all_gains;
features
save features.mat features
R=-2*all_pre./all_gains
L=1./all_gains
C=all_gains./(all_pim.^2+all_pre.^2)

function out=set_imag_part_pos(in)
out=[];
for i=1:length(in)
    if imag(in(i))<0
        in(i)=real(in(i))-1i*imag(in(i));
    end
    out(end+1)=in(i);
end
end

function [Ha, y] = continuous_prony(x,nq,np)
global t
dt=t(2)-t(1);
[b,a]=prony(x, nq, np); 
Hd=tf(b,a,dt);
% use tustin transform. Reduces precision of location so needs enough sampling. zoh/foh has its problems (see literature)
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