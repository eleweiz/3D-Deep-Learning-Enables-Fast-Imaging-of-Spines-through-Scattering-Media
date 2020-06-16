%% The code is to stick mutliple patches;
%% Written by Zhun Wei at Harvard on 26/07/2019;


%% Ground truth load
load ('Results_GroundTruth.mat');
[NN, N_x, N_y, N_z]=size(O_CNN);  % The dimensions of the patches
NNN_x=512;  % The Nx dimensions of the whole image
nn_z=round((NNN_x/N_x)); % number of patters generated for each cell
Gt_L=zeros(NNN_x,NNN_x,N_z,'single');
n1_tem=round(N_x/2);
D=64;
D_loop=N_z/D;
for nn=1:nn_z  % loop for y-axis generate random patterns
    for mm=1:nn_z % loop for x-axis generate random patterns
        NN_n=(nn-1)*4+mm;
        Ind_x=n1_tem+(mm-1)*N_x;
        Ind_y=n1_tem+(nn-1)*N_x;
        Ind_xr=Ind_x-round(N_x/2)+1:Ind_x-round(N_x/2)+N_x;  % Inx range
        Ind_yr=Ind_y-round(N_y/2)+1:Ind_y-round(N_y/2)+N_y;
        Gt_L(Ind_xr,Ind_yr,:)=(GT_CNN(NN_n,:,:,:));
        
    end
end
clearvars -except Gt_L

%% Neural Network Input and Output
clc;close all;
load ('Results_InputOutput.mat');
[NN, N_x, N_y, N_z]=size(O_CNN);  % The dimensions of the patches
NNN_x=512*2;  % The Nx dimensions of the whole image
nn_z=round((NNN_x/N_x)); % number of patters generated for each cell
In_L=zeros(NNN_x,NNN_x,N_z,'single');
Out_L=zeros(NNN_x,NNN_x,N_z,'single');
n1_tem=round(N_x/2);

D=64;
D_loop=N_z/D;
for nn=1:nn_z  % loop for y-axis generate random patterns
    for mm=1:nn_z % loop for x-axis generate random patterns
        NN_n=(nn-1)*nn_z+mm;
        Ind_x=n1_tem+(mm-1)*N_x;
        Ind_y=n1_tem+(nn-1)*N_x;
        Ind_xr=Ind_x-round(N_x/2)+1:Ind_x-round(N_x/2)+N_x;  % Inx range
        Ind_yr=Ind_y-round(N_y/2)+1:Ind_y-round(N_y/2)+N_y;
        In_L(Ind_xr,Ind_yr,:)=(I_CNN(NN_n,:,:,:));
        Out_L(Ind_xr,Ind_yr,:)=(O_CNN(NN_n,:,:,:));       
    end
end

In_Lm=max(In_L,[],3);
Out_Lm=(max(Out_L,[],3));
GT_Lm=max(Gt_L,[],3);
figure; nn_p=100;

subplot(1,3,1);
imshow(GT_Lm,[1,nn_p]); title('GroundTruth');
colormap gray
subplot(1,3,2);
imshow(In_Lm,[1,nn_p]); title('Input');
colormap gray
subplot(1,3,3);
imshow(Out_Lm,[1,nn_p]); title('Output');
colormap gray

In_x=1:1024; In_y=1:1024;nz=52;
In_t1=In_L(In_x,In_y,nz); Out_t1=Out_L(In_x,In_y,nz); GT_t1=Gt_L(:,:,nz); 
figure;
nn_p=85;nn_p1=1;
subplot(1,3,1);
imshow(squeeze(GT_t1),[0,20]);colormap gray; title('GroundTruth');
subplot(1,3,2);
imshow(squeeze(In_t1),[nn_p1,nn_p]);colormap gray; title('Input');
subplot(1,3,3);
imshow(squeeze(Out_t1),[nn_p1,nn_p]);colormap gray; title('Output');








