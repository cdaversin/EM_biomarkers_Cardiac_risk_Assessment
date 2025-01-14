load('BIOMT_3D.mat') 
BIOMT=BIOMT_3D; 

%% Data filtration

outpath=strcat(mfilename('fullpath'),'\');
save_name= strcat(outpath,'BIOMT_clear');
APD0 = find(BIOMT(:,5)==0); %Find sims with APD90=0 
TRI0 = find(BIOMT(:,6)<0); %Find sims with TRI<0: ONE: row 2309 (M29 Ibutilide) --> BAD APD90 MEASURE
VPEAK0 = find(BIOMT(:,7)<0); %Find sims with Vpeak<0: To remove it from the classification
CV0 = find(BIOMT(:,47)<0); %Find sims with Conduction Velocity<0 
BIOMT(APD0,:)=NaN; %NaN to clasify as High TdP-Risk
BIOMT(TRI0,:)=NaN;
BIOMT(VPEAK0,:)=0; %Simulations with Vp<0 are set to zero to eliminate them from the classification
BIOMT(CV0,:)=NaN;


%% Data Normalization
nanRows = any(isnan(BIOMT), 2);
nanRowIndex = find(nanRows);
BIOMT_noNaN = BIOMT(~nanRows,:);
BIOMT_norm_noNaN = zscore(BIOMT_noNaN); 
BIOMT_norm = NaN(size(BIOMT));
BIOMT_norm(~nanRows,:)=BIOMT_norm_noNaN;

BIOMT=BIOMT_norm;
save(save_name,'BIOMT','biom_order','drugs_order')

%% Biomarkers choicing
clear
load('D:\00_FARMACOS\02_SCT_sims\00_BIOM_DAT\51_biom\BIOMT_clear_norm.mat')
outpath='D:\00_FARMACOS\02_SCT_sims\00_BIOM_DAT\01_normalize\5_biom_FS_maxTa_ampTa\';
save_name= strcat(outpath,'5_biom_FS_maxTa_ampTa');

n_biomarker=[5,51,50,13,6]; %5_biom_Ta: APD90, qNet, EMw_Ta, CaTD80, triangulation

%Name and biomarker information
biom_order=biom_order(n_biomarker);
BIOMT=BIOMT(:,n_biomarker);

% Save X biomarkers
save(save_name,'BIOMT','biom_order')



%% Training and validation split
pop=190; %poblation
%training_drugs
%FOR 3D
bep=BIOMT((pop*3+1):pop*4,:); %High
dof=BIOMT((pop*10+1):pop*11,:); %High
quin=BIOMT((pop*21+1):pop*22,:); %High
sot=BIOMT((pop*24+1):pop*25,:); %High
chlor=BIOMT((pop*4+1):pop*5,:); %Intermediate
cisa=BIOMT((pop*5+1):pop*6,:); %Intermediate
ondan=BIOMT((pop*19+1):pop*20,:); %Intermediate
terf=BIOMT((pop*26+1):pop*27,:); %Intermediate
dilt=BIOMT((pop*8+1):pop*9,:); %Low
mexi=BIOMT((pop*16+1):pop*17,:); %Low
ranol=BIOMT((pop*22+1):pop*23,:); %Low
vera=BIOMT((pop*28+1):pop*29,:); %Low
 
% %FOR 0D
% bep=BIOMT((pop*2+1):pop*3,:); %High
% dof=BIOMT((pop*9+1):pop*10,:); %High
% quin=BIOMT((pop*20+1):pop*21,:); %High
% sot=BIOMT((pop*23+1):pop*25,:); %High
% chlor=BIOMT((pop*3+1):pop*4,:); %Intermediate
% cisa=BIOMT((pop*4+1):pop*5,:); %Intermediate
% ondan=BIOMT((pop*18+1):pop*19,:); %Intermediate
% terf=BIOMT((pop*25+1):pop*26,:); %Intermediate
% dilt=BIOMT((pop*7+1):pop*8,:); %Low
% mexi=BIOMT((pop*15+1):pop*16,:); %Low
% ranol=BIOMT((pop*21+1):pop*22,:); %Low
% vera=BIOMT((pop*27+1):pop*28,:); %Low

train_drug_names={'Bepridil','Dofetilide','Quinidine','Sotalol','Chlorpromazine','Cisapride','Ondansetron','Terfenadine','Diltiazem','Mexiletine','Ranolazine','Verapamil'};
train_BIOMT = vertcat(bep,dof,quin,sot,chlor,cisa,ondan,terf,dilt,mexi,ranol,vera);
%train_BIOMT=table2array(vertcat(bep,dof,quin,sot,chlor,cisa,ondan,terf,dilt,mexi,ranol,vera)); %FOR 0D
clear bep dof quin sot chlor cisa ondan terf dilt mexi ranol vera
save(strcat(outpath,'train_BIOMT'), 'train_BIOMT','train_drug_names', 'biom_order');

%test_drugs
%FOR 3D
azim=BIOMT((pop*2+1):pop*3,:);
diso=BIOMT((pop*9+1):pop*10,:);
ibut=BIOMT((pop*13+1):pop*14,:);
vande=BIOMT((pop*27+1):pop*28,:);
astem=BIOMT((pop*1+1):pop*2,:);
clari=BIOMT((pop*6+1):pop*7,:);
cloz=BIOMT((pop*7+1):pop*8,:);
dompe=BIOMT((pop*11+1):pop*12,:);
drop=BIOMT((pop*12+1):pop*13,:);
pimo=BIOMT((pop*20+1):pop*21,:);
rispe=BIOMT((pop*23+1):pop*24,:);
lora=BIOMT((pop*14+1):pop*15,:);
meto=BIOMT((pop*15+1):pop*16,:);
nife=BIOMT((pop*17+1):pop*18,:);
nitren=BIOMT((pop*18+1):pop*19,:);
tamo=BIOMT((pop*25+1):pop*26,:);

%FOR 0D
% azim=BIOMT((pop*1+1):pop*2,:);
% diso=BIOMT((pop*8+1):pop*9,:);
% ibut=BIOMT((pop*12+1):pop*13,:);
% vande=BIOMT((pop*26+1):pop*27,:);
% astem=BIOMT((pop*0+1):pop*1,:);
% clari=BIOMT((pop*5+1):pop*6,:);
% cloz=BIOMT((pop*6+1):pop*7,:);
% dompe=BIOMT((pop*10+1):pop*11,:);
% drop=BIOMT((pop*11+1):pop*12,:);
% pimo=BIOMT((pop*19+1):pop*20,:);
% rispe=BIOMT((pop*22+1):pop*23,:);
% lora=BIOMT((pop*13+1):pop*14,:);
% meto=BIOMT((pop*14+1):pop*15,:);
% nife=BIOMT((pop*16+1):pop*17,:);
% nitren=BIOMT((pop*17+1):pop*18,:);
% tamo=BIOMT((pop*24+1):pop*25,:);

val_drug_names={'Azimilide','Disopyramide','Ibutilide','Vandetanib','Astemizole','Clarithromycin','Clozzpine','Domperidone','Droperidol','Pimozide','Risperidone','Loratadine','Metoprolol','Nifedipine','Nitrendipine','Tamoxifen'};
%val_BIOMT = table2array(vertcat(azim,diso,ibut,vande,astem,clari,cloz,dompe,drop,pimo,rispe,lora,meto,nife,nitren,tamo)); %FOR 0D
val_BIOMT = vertcat(azim,diso,ibut,vande,astem,clari,cloz,dompe,drop,pimo,rispe,lora,meto,nife,nitren,tamo); %FOR 3D
clear azim diso ibut vande astem clari cloz dompe drop pimo rispe lora meto nife nitren tamo
save(strcat(outpath,'val_BIOMT'), 'val_BIOMT',"val_drug_names",'biom_order');

%% Feature selection-MRMR
pop_size= 190;
feat_selected=5;
y_training=[1*ones(pop_size*4,1); 2*ones(pop_size*4,1); 3*ones(pop_size*4,1)];
rows_with_Vp = all(train_BIOMT == 0, 2); %Index of rows with Vp<0
train_BIOMT=train_BIOMT(~rows_with_Vp,:);
y_training=y_training(~rows_with_Vp);
[idx,scores] = fsrmrmr(train_BIOMT,y_training);

figure
b= bar(scores(idx),'FaceColor',[0 .5 .5]);
b.FaceColor = 'flat';
for i=feat_selected+1:length(scores)
b.CData(i,:) = [0 0 0];
end
xticklabels(strrep(biom_order(idx),'_','\_'))
xlabel('Biomarker rank')
ylabel('Biomarker importance score')
title('MRMR feature selection algorithm')

%% Feature selection-Correlation
clear
pop_size = 190;
y_training=[1*ones(pop_size*4,1); 2*ones(pop_size*4,1); 3*ones(pop_size*4,1)];
%Delete rows with Vp>0
rows_with_Vp = all(train_BIOMT == 0, 2); %Index of rows with Vp<0
train_BIOMT=train_BIOMT(~rows_with_Vp,:);
y_training=y_training(~rows_with_Vp);

%Calculate correlation coeficients
corr = corrcoef([train_BIOMT,y_training],'Rows','complete');
%Higher correlations >0.95 in features 5,6,7 --> Delete Catd50 and CaTD90 

%Calculate ttest
for i=1:size(train_BIOMT,2)
[h(i),p(i)] = ttest2(train_BIOMT(:,i),y_training);
end