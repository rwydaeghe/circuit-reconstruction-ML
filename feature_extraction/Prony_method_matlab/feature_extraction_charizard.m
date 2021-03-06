clearvars; close all; delete(findall(0,'type','figure','tag','TMWWaitbar'));
warning('off','all')
wb = waitbar(0,'Loading transient data...');
global t
charizard=load('charizard_transient_data_set');
circuits=fieldnames(charizard);
N_circuits=length(circuits);
features=struct;
plotting_on=false; %you most definitely don't want this for big data sets
for i_circuit=1:N_circuits
    circuit=circuits(i_circuit);
    t_and_data=charizard.(circuit{1});
    t=t_and_data(:,1);
    data=t_and_data(:,2:end);
    N_transients=length(data(1,:)); %number of varied values versions of the circuit
    ideal_np_found=false; ideal_nq_found=false;
    best_np=[]; best_nq=[];
    for i_transient=1:N_transients
        x_i=data(:,i_transient);
        max_np=2*4; %2x the expected amount. arbitrary. seems to give best results.
        err_best=inf;
        err_plot=[]; nq_plot=[]; np_plot=[];
        %flatline detection on small samples
        if length(best_np)>1
            if sum(best_np==median(best_np))/length(best_np)>.6
                ideal_np_found=true;
                ideal_np=median(best_np);
            elseif ideal_np_found==true
                disp('Serious misjudgment of amount of poles happened for i_transient='+num2str(i_transient)+' and i_circuit='+num2str(i_circuit))
            end
        end
        if length(best_nq)>1
            if sum(best_nq==median(best_nq))/length(best_nq)>.6
                ideal_nq_found=true;
                ideal_nq=median(best_nq);
            elseif ideal_np_found==true
                disp('Serious misjudgment of amount of zeros happened for i_transient='+num2str(i_transient)+' and i_circuit='+num2str(i_circuit)) 
            end
        end
        if i_transient==1 && i_circuit==1
            progress_step=1/(N_circuits*N_transients);
            progress=-progress_step; % so it starts at 0% and finishes at 100%
        end
        progress=progress+progress_step; 
        sub_progress_step=1/(max_np*(max_np+1)/2);
        sub_progress=0;
        if ideal_np_found==false && ideal_nq_found==false            
            for np=1:max_np
                for nq=1:np
                    sub_progress=sub_progress+sub_progress_step;
                    real_progress=progress+sub_progress*progress_step;
                    waitbar(real_progress,wb,...
                        "Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+...
                        ", transient "+num2str(i_transient)+"/"+num2str(N_transients)+...
                        " [Best np & nq not found]... ("+sprintf("%0.2f", real_progress*100)+"%)");

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
            best_np(end+1)=np;
            best_nq(end+1)=nq;
        else
            np=ideal_np;
            nq=ideal_nq;
            real_progress=progress;
            waitbar(real_progress,wb,...
                "Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+...
                ", transient "+num2str(i_transient)+"/"+num2str(N_transients)+...
                " [Best np & nq found]... ("+sprintf("%0.2f", real_progress*100)+"%)");

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
        if plotting_on
            figure(2*i_transient-1)
            hold on;
            ylim([min(x_i)*1.5, max(x_i)*1.5])
            plot(t,x_i)
            plot(t,y_best)
        end
        % error evolution plot along with nq, np
        if plotting_on
            figure(2*i_transient)
            hold on;
            loglog(1:length(err_plot), err_plot)
            for i_transient=1:length(err_plot)
                text(i_transient, err_plot(i_transient), "("+nq_plot(i_transient)+","+np_plot(i_transient)+")")
            end
        end
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(i_transient))).zeros=zero(Ha_best);
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(i_transient))).poles=pole(Ha_best);
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(i_transient))).gain=Ha_best.K;
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(i_transient))).error=err_best;
    end
    amount_of_zeros=zeros(1,N_transients);
    amount_of_poles=zeros(1,N_transients);
    for j=1:N_transients
        amount_of_zeros(j)=length(datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(j))).zeros);
        amount_of_poles(j)=length(datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(j))).poles);
    end
    amount_of_zeros=ceil(median(amount_of_zeros));
    amount_of_poles=ceil(median(amount_of_poles));
    all_zeros=zeros(amount_of_zeros,N_transients);
    all_poles=zeros(amount_of_poles,N_transients);
    all_gains=zeros(1,N_transients);
    all_errors=zeros(1,N_transients);
    for k=1:N_transients
        if length(datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).zeros)==amount_of_zeros
            all_zeros(:,k)=datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).zeros;
        else
            all_zeros(:,k)=nan;
        end
        if length(datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).poles)==amount_of_poles
            all_poles(:,k)=datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).poles;
        else
            all_poles(:,k)=nan;
        end
        all_gains(1,k)=datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).gain;
        all_errors(1,k)=datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('transient_',num2str(k))).error;
    end
    all_pre_with_doubles=1/2*(all_poles+conj(all_poles));
    all_pim_with_doubles=1i/2*((all_poles-conj(all_poles))); %positive imag part
    all_zre_with_doubles=1/2*(all_zeros+conj(all_zeros));
    all_zim_with_doubles=1i/2*((all_zeros-conj(all_zeros))); %positive imag part
    
    % get the doubles out of these arrays
    all_pre=[]; all_pim=[]; 
    all_zre=[]; all_zim=[]; 
    for l=1:N_transients
        all_pre_l=[]; all_pim_l=[]; 
        if isempty(all_poles)==false
            skip_next_one=false;
            for p=[all_pre_with_doubles(:,l),all_pim_with_doubles(:,l)]'
                if skip_next_one==true
                    skip_next_one=false;
                    continue
                end
                if p(2)==0 %it doesn't come as a double                    
                    all_pre_l(end+1)=p(1);
                    all_pim_l(end+1)=0;
                else %it comes as a double                    
                    all_pre_l(end+1)=p(1);
                    all_pim_l(end+1)=abs(p(2)); %it should already be positive anyways
                    skip_next_one=true;
                end
            end
        end
        if isempty(all_pre_l)==false
            if l>1
                if all(size(all_pre(l-1,:))==size(all_pre_l))
                    all_pre(l,:)=all_pre_l;
                else
                    %usually happens when there's the same amount of poles, but one of them has more conjugate pairs
                    all_pre(l,:)=all_pre(l-1,:)*nan; 
                end
            else
                all_pre(l,:)=all_pre_l;
            end
        end
        if isempty(all_pim_l)==false
            if l>1
                if all(size(all_pim(l-1,:))==size(all_pim_l))
                    all_pim(l,:)=all_pim_l;
                else
                    %usually happens when there's the same amount of poles, but one of them has more conjugate pairs
                    all_pim(l,:)=all_pim(l-1,:)*nan;
                end
            else
                all_pim(l,:)=all_pim_l;
            end
        end
        all_zre_l=[]; all_zim_l=[];
        if isempty(all_zeros)==false
            skip_next_one=false;
            for z=[all_zre_with_doubles(:,l),all_zim_with_doubles(:,l)]'
                if skip_next_one==true
                    skip_next_one=false;
                    continue
                end
                if z(2)==0 %it doesn't come as a double
                    all_zre_l(end+1)=z(1);
                    all_zim_l(end+1)=0;
                else %it comes as a double
                    all_zre_l(end+1)=z(1);
                    all_zim_l(end+1)=abs(z(2)); %it should already be positive anyways
                    skip_next_one=true;
                end
            end
        end
        if isempty(all_zre_l)==false
            if l>1
                if all(size(all_zre(l-1,:))==size(all_zre_l))
                    all_zre(l,:)=all_zre_l;
                else
                    %usually happens when there's the same amount of zeros, but one of them has more conjugate pairs
                    all_zre(l,:)=all_zre(l-1,:)*nan;
                end
            else
                all_zre(l,:)=all_zre_l;
            end
        end
        if isempty(all_zim_l)==false
            if l>1
                if all(size(all_zim(l-1,:))==size(all_zim_l))
                    all_zim(l,:)=all_zim_l;
                else
                    %usually happens when there's the same amount of zeros, but one of them has more conjugate pairs
                    all_zim(l,:)=all_zim(l-1,:)*nan;
                end
            else
                all_zim(l,:)=all_zim_l;
            end
        end
    end
    features.(strcat('circuit_',num2str(i_circuit))).zeros_re=all_zre;
    features.(strcat('circuit_',num2str(i_circuit))).zeros_im=all_zim;
    features.(strcat('circuit_',num2str(i_circuit))).poles_re=all_pre;
    features.(strcat('circuit_',num2str(i_circuit))).poles_im=all_pim;
    features.(strcat('circuit_',num2str(i_circuit))).gains=all_gains;    
end
waitbar(1, wb, 'Saving feature data...')
save charizard_features.mat features
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