% function genTrData(sl_em,z,exp_t_factor,rsf)

clc;
clear all;
close all;
format short;
%% parameter definition
sl_em = 65; % mScarlet-I: Measurement of multispectral scattering
% sl_em = 44; %EYFP

exp_t_factor = 3;  % match the power
% addpath(genpath('./supportingFunctions/'))  % function path
dataPath_root = './_data/20190317_SOM mice/'; % data root
files = dir([dataPath_root '**/*.tif*']);

dataSetDir = './_trData/';
InDir = fullfile(dataSetDir,'In');
OutDir = fullfile(dataSetDir,'Out');

z_temp1=64; % Start point of z
z_temp2=1; % End point of z
NN_x=128;NN_y=128;NN_z=z_temp1-z_temp2+1;
NN_t=round(length(files)*130); % total number of data
NN_p=round(250/170*800); %total number of pixels
dx_gt = (0.25*800)/NN_p;% [um] pixel size
dx = dx_gt;   % [um] TF image's pixel size
Nx = NN_p;
nn_z=round(NN_t/(length(files)-1)); % number of patters generated for each cell
Count_nn_tem=1;
np=100;
inx_tem1=100;% boundaries of the inx minmum and maxium;
% R_power=(0.66*8.47)/(18*0.04);


R_power=1;
for i=1:length(files)-1
    fname = fullfile(files(i).folder,files(i).name);
    info = imfinfo(fname);
    Nz = numel(info);
    kk=1;  % the z dimension
    z=z_temp2  % The initialization of z
    for mm=64:-1:1
        if i<28  % for data having 1 channel 
            j=mm;  % loop for depth
        else      % for data having 3 channels
            j=(mm-1)*3+1; % loop for depth
        end
        sPSF = sim_get_modeled_sPSF(z,sl_em,dx,round(0.5*Nx));   % [um] simulated PSF
        
        I_temp1 = single(imread(fname,j));  % Red channel
        I_temp2 = imresize(I_temp1,[NN_p NN_p]);
        I_temp=exp_t_factor*round(I_temp2);  % match power in exp
        
        I_temp3=R_power*I_temp;
        
        J_temp = conv2(I_temp3,sPSF,'same');  % scale magnification
        J_temp(J_temp<0)=0;
        
        J_temp = poissrnd(J_temp);
        J_temp= 1/R_power*J_temp;
        J_temp=round(J_temp);
        
        I_temp(I_temp<0)=0;
        I_temp(I_temp>np)=np;
        
        J_temp(J_temp<0)=0;
        J_temp(J_temp>np)=np;
        
        [x_inx,y_inx]=find(I_temp>0.2*np);
        Inx_tem=(x_inx>inx_tem1).*(y_inx>inx_tem1).*(x_inx<NN_p-inx_tem1).*(y_inx<NN_p-inx_tem1);
        AA=find(Inx_tem>0.1);
        x_inx_rand=x_inx(AA);
        y_inx_rand=y_inx(AA);
        len=length(x_inx_rand)-1; % Avoid zero indx
%                 figure; imagesc((I_temp)); colormap hot; colorbar;title('GT')
%                 figure; imagesc((J_temp)); colormap hot; colorbar;title('In')
%                 return
        for nn=Count_nn_tem:Count_nn_tem+nn_z-1  % loop for generate random patterns
            rand('state', nn);
            Ind_x=x_inx_rand(round(len*rand)+1);
            rand('state', nn);
            Ind_y=y_inx_rand(round(len*rand)+1);
            Ind_xr=Ind_x-round(NN_x/2):Ind_x-round(NN_x/2)+NN_x-1;  % Inx range
            Ind_yr=Ind_y-round(NN_y/2):Ind_y-round(NN_y/2)+NN_y-1;
            temp_1=I_temp(Ind_xr,Ind_yr);
            temp_2=J_temp(Ind_xr,Ind_yr);
            
            kk
            nn
            I_out(kk,:,:,nn)=single((temp_1).'); % for h5py loading in python
            I_in(kk,:,:,nn)=single((temp_2).');  % for h5py loading in python
            %             figure; imagesc(squeeze((I_out(kk,:,:,nn)))); colormap hot; colorbar;title('GT')
            %             figure; imagesc(squeeze((I_in(kk,:,:,nn)))); colormap hot; colorbar;title('In'); return
        end
        kk=kk+1;
        if kk>NN_z
            Count_nn_tem=nn+1;  % if all z direction has counted, then start count at maximum point
        end
        z = z+1
    end
    
end
x_r=round(rand*10); x_h=round(rand*3000);
% display one example
figure; imagesc(squeeze((I_out(x_r,:,:,x_h)))); colormap hot; colorbar;title('GT')
figure; imagesc(squeeze((I_in(x_r,:,:,x_h)))); colormap hot; colorbar;title('In')
M_in=min(min(min(min(I_in))));
M_out=min(min(min(min(I_out))));
M_coef=min([M_in,M_out]);
I_in=I_in-M_coef;
I_out=I_out-M_coef;
save('Training_SpineN_in','I_in')
save('Training_SpineN_out','I_out')


clear all;













