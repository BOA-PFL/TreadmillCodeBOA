clear
clc
clf

% Data files should be named Subject_Shoe_Movement_Timepoint - Forces
%input_dir = 'C:\Users\Daniel.Feeney\Dropbox (Boa)\Endurance Protocol Trail Run\Michel_LaSportiva\RunData';% Change to correct filepath
input_dir = 'C:\Users\kate.harrison\Dropbox (Boa)\EndurancePerformance\NewBalanceRC_May2021\Forces';
direction = 1; % If level or uphill = 1 (forwards); If downhill = 2 (backwards)

cd(input_dir)
files = dir('*Forces.txt');
dataList = {files.name};
dataList = sort(dataList);
[f,~] = listdlg('PromptString','Select data files for all subjects in group','SelectionMode','multiple','ListString',dataList);
NumbFiles = length(f);

subject = cell(3,1);
config = cell(3,1);
%period = cell(3,1);
VILR = zeros(3,1);
VALR = zeros(3,1);
RILR = zeros(3,1);
RALR = zeros(3,1);
pBF = zeros(3,1);
pLF = zeros(3,1);
pMF = zeros(3,1);
contactTime = zeros(3,1);
row = 0;

for s = 1:NumbFiles

    fileName = dataList{f(s)};
    fileLoc = [input_dir '\' fileName]; 
    names = split(fileName, ["_"," "]);
    sub = names(1);
    conf = names(3);
    %p = names(4);
    M = dlmread(fileLoc, '\t', 9, 0);
    M(isnan(M)) = 0;
    
    Fz = M(:,4)*-1;%vertical
    %LankleForce = M(:,13);
    %LankleMoment = M(:,14);
    
    if direction == 1
    Fy = -1.*M(:,5);%antero-posterior
    Fx = -1.*M(:,6);%mediolateral
    else
    Fy = M(:,5);%antero-posterior
    Fx = M(:,6);%mediolateral
    end
    
    
    %LankleForce = filtfilt(b,a,LankleForce);
    %LankleMoment = filtfilt(b,a,LankleMoment);
    %LmuscleForce = LankleMoment ./ 0.05; 
    %LtibForce = LankleForce + LankleMoment;
    
% hold off
% plot(LankleMoment)
% title(fileName)

start = 100;
    if start < 5000
        dFz = diff(Fz);
    
        Fz(Fz<80) = 0;
        dFz(Fz<80) = 0;
        Fy(Fz<80) = 0;
        Fx(Fz<80) = 0;
        Fr = sqrt(Fz.^2 + Fy.^2 + Fx.^2);
        dFr = diff(Fr);

   
        contact = zeros(length(Fz),1);
        off = zeros(length(Fz),1);
        
         for i = 2:length(Fz)
            if (Fz(i-1)<80) && (Fz(i)>80)
            contact(i)=1;
            end
         end
    
        for i = 1:length(Fz)-1
            if (Fz(i)>80) && (Fz(i+1)<80)
            off(i) = 1;
            end
        end
    
        ic = find(contact==1);
        to = find(off==1);
        spm = length(ic);
    
        if ~isempty(ic)
            
            if to(1)<ic(1)
                to = to(2:end); % remove any toe off before first IC
            end
    
            ic = ic(1:length(to)); % make sure there is a toe off for every ic
    
            % Is first step right or left?
    
            footLoc1 = mean(M(ic(1):to(1),2));
            footLoc2 = mean(M(ic(2):to(2),2));
    
            if s == 1 
            idx = 1:2:length(ic);
            Ric = ic(idx);
            Rto = to(idx);
        
            elseif footLoc1 > footLoc2
            idx = 2:2:length(ic);
            Ric = ic(idx);
            Rto = to(idx);
        
            else 
            idx = 1:2:length(ic);
            Ric = ic(idx);
            Rto = to(idx);
        
            end
    
            for i = 1:length(Ric)
                
            stepLen = Rto(i) - Ric(i);
            pct20 = Ric(i) + round(stepLen*0.03);
            pct80 = Ric(i) + round(stepLen*0.12);
            
            Zval20 = Fz(pct20);
            Zval80 = Fz(pct80);
            
            Rval20 = Fr(pct20);
            Rval80 = Fr(pct80);
                   
                           
            VALR(row + i) = (Zval80-Zval20)/(pct80-pct20);
            VILR(row + i) = max(dFz(Ric(i):Rto(i)));
            RALR(row + i) = (Rval80-Rval20)/(pct80-pct20);
            RILR(row + i) = max(dFr(Ric(i):Rto(i)));
            pBF(row + i) = min(Fy(Ric(i):Rto(i)));
            pLF(row + i) = max(Fx(Ric(i):Rto(i)));
            pMF(row + i) = max(-1*Fx(Ric(i):Rto(i)));
            contactTime(row + i) = stepLen/1000;
      

            config(row + i) = conf;
%             period(row + i) = p;
            subject(row + i) = sub;
            end 
       end 
    end 
        
    row = length(VILR);
end
        
  
  ColTitles = {'SubjectName', 'ShoeCondition','VALR', 'VILR', 'RALR', 'RILR', 'pBF', 'pMF', 'pLF', 'contactTime'};
  KinData = horzcat(VALR, VILR, RALR, RILR, pBF, pMF, pLF, contactTime);
  KinData = num2cell(KinData);
  KinData = horzcat(subject,config,KinData);
  KinData = vertcat(ColTitles, KinData);
  writecell(KinData, 'CompiledRunData.csv')
  
