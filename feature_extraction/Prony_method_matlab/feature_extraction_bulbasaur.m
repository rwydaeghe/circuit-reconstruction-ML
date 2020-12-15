clearvars; close all; delete(findall(0,'type','figure','tag','TMWWaitbar'));
warning('off','all')
wb = waitbar(0,'Loading transient data...');
global t
bulbasaur=load('bulbasaur_transient_data_set');
circuits=fieldnames(bulbasaur);
N_circuits=length(circuits);
features=struct;
plotting_on=true; %you most definitely don't want this for big data sets
print_progresses=true; 
progress_step=1/N_circuits;
progress=-progress_step; % so it starts at 0% and finishes at 100%
for i_circuit=1:N_circuits
    circuit=circuits(i_circuit);
    t_and_data=bulbasaur.(circuit{1});
    t=t_and_data.('t');
    nodes=fieldnames(t_and_data);
    N_nodes=length(nodes)-1; %number of nodes in the circuit (starts at 2)
    ideal_np_found=false; best_np=[]; 
    progress=progress+progress_step;
    sub_progress_step=1/N_nodes;
    sub_progress=-sub_progress_step;
    for i_node=1:N_nodes
        node=nodes(i_node+1);
        x_i=t_and_data.(node{1}).';
        max_np=2*4; %2x the expected amount. arbitrary. seems to give best results.
        err_best=inf;
        err_plot=[]; np_plot=[]; nq_plot=[];
        %flatline detection on small samples
        if length(best_np)>1
            if sum(best_np==median(best_np))/length(best_np)>.6
                ideal_np_found=true;
                ideal_np=median(best_np);
            elseif ideal_np_found==true
                disp('Serious misjudgment of amount of poles happened for i_node='+num2str(i_node+1)+' and i_circuit='+num2str(i_circuit))
            end
        end
        sub_progress=sub_progress+sub_progress_step;
        if ideal_np_found==false
            sub_sub_progress_step=1/(max_np*(max_np+1)/2);
            sub_sub_progress=-sub_sub_progress_step;
            for np=1:max_np
                for nq=1:np
                    sub_sub_progress=sub_sub_progress+sub_sub_progress_step;
                    real_progress=progress+sub_progress*progress_step+sub_sub_progress*sub_progress_step*progress_step;
                    if print_progresses
                        disp("Progress="+num2str(progress))
                        disp("Sub_progress="+num2str(sub_progress))
                        disp("Sub_sub_progress="+num2str(sub_sub_progress))
                        disp("Real progress="+num2str(real_progress))
                        disp("----------------")
                    end
                    waitbar(real_progress,wb,...
                        "Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+...
                        ", node "+num2str(i_node)+"/"+num2str(N_nodes)+...
                        " [np not found]... ("+sprintf("%0.2f", real_progress*100)+"%)");

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
        else
            np=ideal_np;
            sub_sub_progress_step=1/np;
            sub_sub_progress=-sub_sub_progress_step;            
            for nq=1:np
                sub_sub_progress=sub_sub_progress+sub_sub_progress_step;
                real_progress=progress+sub_progress*progress_step+sub_sub_progress*sub_progress_step*progress_step;
                if print_progresses
                    disp("Progress="+num2str(progress))
                    disp("Sub_progress="+num2str(sub_progress))
                    disp("Sub_sub_progress="+num2str(sub_sub_progress))
                    disp("Real progress="+num2str(real_progress))
                    disp("----------------")
                end
                waitbar(real_progress,wb,...
                    "Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+...
                    ", node "+num2str(i_node)+"/"+num2str(N_nodes)+...
                    " [np found]... ("+sprintf("%0.2f", real_progress*100)+"%)");

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
            figure(2*i_node-1)
            clf
            title("Actual and predicted signals. Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+", node "+num2str(i_node)+"/"+num2str(N_nodes))
            hold on;
            mi=min(x_i); ma=max(x_i);
            if ~(mi==0 && ma==0)
                ylim([mi*1.5, ma*1.5])
            end
            plot(t,x_i)
            plot(t,y_best)
        end
        % error evolution plot along with nq, np
        if plotting_on
            figure(2*i_node)
            clf
            title("Log10 error as a function of (nq,np). Circuit "+num2str(i_circuit)+"/"+num2str(N_circuits)+", node "+num2str(i_node)+"/"+num2str(N_nodes))
            hold on;
            plot(1:length(err_plot), log10(err_plot))
            for e_plot_i=1:length(err_plot)
                text(e_plot_i, log10(err_plot(e_plot_i)), "("+nq_plot(e_plot_i)+","+np_plot(e_plot_i)+")")
            end
        end
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).zeros=zero(Ha_best);
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).poles=pole(Ha_best);
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).gain=Ha_best.K;
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).err_plot=err_plot;
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).nq_plot=nq_plot;
        datastruct.(strcat('circuit_',num2str(i_circuit))).(strcat('node_',num2str(i_node+1))).np_plot=np_plot;
    end 
end
waitbar(1, wb, 'Saving feature data...')
save bulbasaur_features.mat datastruct
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
gain=x(10)/y(10); % this works due to linearity of laplace tf. hope that y point isnt zero (to do). choose cell 10 because usually not zero (or <eps) at the beginning
if abs(y(10))<eps
    disp('Potentially a problem in gain evaluation happened...')
    if ~any(y)
        %it's completely zero
        disp("prony signal it's completely zero")
    else
        disp("y(10)="+num2str(y(10)))
    end
    zeros
    poles
    gain
end
Ha=zpk(zeros,poles,gain);
y=y*gain; %ditto, no double calculations necessary :)
end    