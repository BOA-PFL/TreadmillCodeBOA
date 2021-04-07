clear
clc
clf

% Data files should be named Subject_Shoe_Movement_Timepoint - Forces
%input_dir = 'C:\Users\Daniel.Feeney\Dropbox (Boa)\Endurance Protocol Trail Run\Michel_LaSportiva\RunData';% Change to correct filepath
input_dir = 'C:\Users\Daniel.Feeney\Dropbox (Boa)\FBS Abstract\Running GRF';
direction = 1; % If level or uphill = 1 (forwards); If downhill = 2 (backwards)

% line = {'-k','-g', '-b','-r'}; % Add enough line colors for each shoe.
% HR = [125.3777778
% 160
% 163.85
% 164.943662
% ];
% Met =[958.5401333
% 1290.120621
% 1309.474175
% 1287.524901
% ];

cd(input_dir)
files = dir('*Forces.txt');
dataList = {files.name};
dataList = sort(dataList);
[f,~] = listdlg('PromptString','Select data files for all subjects in group','SelectionMode','multiple','ListString',dataList);
NumbFiles = length(f);

subject = cell(3,1);
config = cell(3,1);
period = cell(3,1);
VILR = zeros(3,1);
VALR = zeros(3,1);
pBF = zeros(3,1);
pPF = zeros(3,1);
pVGRF = zeros(3,1);
pLF = zeros(3,1);
pMF = zeros(3,1);
BimpTot = zeros(3,1);
PimpTot = zeros(3,1);
MimpTot = zeros(3,1);
LimpTot = zeros(3,1);
row = 0;

for s = 1:NumbFiles

    fileName = dataList{f(s)};
    fileLoc = [input_dir '\' fileName]; 
    names = split(fileName, ["_"," "]);
    sub = names(1);
    conf = names(2);
    p = names(4);
    M = dlmread(fileLoc, '\t', 9, 0);
    M(isnan(M)) = 0;
    
    Fz = M(100:4000,4);%vertical
    
    if direction == 1
    Fy = -1.*M(100:4000,5);%antero-posterior
    Fx = -1.*M(100:4000,6);%mediolateral
    else
    Fy = M(100:4000,5);%antero-posterior
    Fx = M(100:4000,6);%mediolateral
    end
    
    
    fq = 100; %Sampling frequency
    fc = 30; %Cutoff frequency
    
    [b,a] = butter(2,fc/(fq/2),'low'); %second order BW filter, filtfilt doubles the order
    Fz = filtfilt(b,a,Fz); Fz = (Fz * -1);
    Fy = filtfilt(b,a,Fy);
    Fx = filtfilt(b,a,Fx);
    
    dFz = diff(Fz);
    
    Fz(Fz<80) = 0;
    dFz(Fz<80) = 0;
    Fy(Fz<80) = 0;
    Fx(Fz<80) = 0;
   
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
    
    FzInt = zeros(length(Ric),101);
    FyInt = zeros(length(Ric),101);
    FxInt = zeros(length(Ric),101);
    
    intframes = 0:100;

%     for i = 1:length(Ric)
%         
%         frames = length(Ric(i):Rto(i));
%         vq = intframes * (frames/101);
%         FzInt(i,:) = interp1(1:frames, Fz(Ric(i):Rto(i)), vq, 'spline');
%         FyInt(i,:) = interp1(1:frames, Fy(Ric(i):Rto(i)), vq, 'spline');
%         FxInt(i,:) = interp1(1:frames, Fx(Ric(i):Rto(i)), vq, 'spline');
%     end
    
%     figure(1)
%     hold on
%     z(s) = shadedErrorBar(0:100, FzInt, {@mean, @std}, 'lineprops', line{s});
%     title('Vertical Force')
%     
%     
%     figure(2)
%     hold on
%     y(s) = shadedErrorBar(0:100, FyInt, {@mean, @std}, 'lineprops', line{s});
%     title('Anteroposterior Force')
%     
%     figure(3)
%     hold on
%     x(s) = shadedErrorBar(0:100, FxInt, {@mean, @std}, 'lineprops', line{s});
%     title('Mediolateral Force (Lateral positive)')

%     plot(dFz(1:1000))
%     hold on
%     plot(Fz(1:1000))
  
    for i=1:length(Ric)
      %%%% Heuristic to get 80% 20% force for VLR %%%
      
      [pk, pkloc] = max(Fz(Ric(i):Rto(i)));
     
     % Using distint impact peak
     
        for m = Ric(i):Ric(i)+pkloc-1
            if (dFz(m)>0) && (dFz(m+1)<0)
            ip(m)=1;
            end
        end
     
        iploc1 = find(ip==1);
        iploc1 = iploc1(1);
     
        % Using impact inflection
        [ip2, ~] = islocalmin(dFz(Ric(1):Ric(10)+pkloc));
        iploc2 = find(ip2==1);
        iploc2 = iploc2(1);
     
        % Using 18% stance time, for cases where there is no impact peak or
        % inflection
        iploc3 = round(length(Fz(Ric(i):Rto(i))*0.18));
     
        % Find the first of these to occur
     
        if isempty(iploc1) == 1 && isempty(iploc2) == 1
            iploc = iploc3;
        elseif isempty(iploc1) == 0 && isempty(iploc2) == 0
            iploc = min(iploc1, iploc2);
        elseif isempty(iploc1) == 1
            iploc = iploc2;
        else
            iploc = iploc1;
        end 
     
        pct20 = Ric(i) + round(0.2*iploc);
        pct80 = Ric(i) + round(0.8*iploc);
     
        val20 = Fz(pct20);
        val80 = Fz(pct80);
               
      VALR(row + i) = (val80-val20)/(pct80-pct20);
      VILR(row + i) = max(dFz(Ric(i):Rto(i)));
      pBF(row + i) = min(Fy(Ric(i):Rto(i)));
      pPF(row + i) = max(Fy(Ric(i):Rto(i)));
      pVGRF(row + i) = max(Fz(Ric(i):Rto(i)));
      pLF(row + i) = max(Fx(Ric(i):Rto(i)));
      pMF(row + i) = max(-1*Fx(Ric(i):Rto(i)));
      
      BFy = Fy;
      BFy(BFy>0) = 0;
      PFy = Fy;
      PFy(PFy<0) = 0;
      LFx = Fx;
      LFx(LFx<0) = 0;
      MFx = Fx;
      MFx(MFx>0) = 0;
  
      Bimp = BFy.*0.001;
      Pimp = PFy.*0.001;
      Mimp = MFx.*0.001;
      Limp = LFx.*0.001;
      
      BimpTot(row + i) = sum(Bimp(Ric(i):Rto(i)));
      PimpTot(row + i) = sum(Pimp(Ric(i):Rto(i)));
      MimpTot(row + i) = sum(Mimp(Ric(i):Rto(i)));
      LimpTot(row + i) = sum(Limp(Ric(i):Rto(i)));
      
      config(row + i) = conf;
      period(row + i) = p;
      subject(row + i) = sub;
      
      
    end 
  row = length(VILR);
end
  
  ColTitles = {'Subject', 'ShoeCondition','Period','VALR', 'VILR', 'pBF', 'pPF', 'pVGRF', 'pMF', 'pLF', 'Bimp', 'Pimp', 'Mimp', 'Limp'};
  KinData = horzcat(VALR, VILR, pBF, pPF, pVGRF, pMF, pLF, BimpTot, PimpTot, MimpTot, LimpTot);
  KinData = num2cell(KinData);
  KinData = horzcat(subject,config,period,KinData);
  KinData = vertcat(ColTitles, KinData);
  writecell(KinData, 'CompiledRunData2.csv')
  
% figure(1)
% legend([z(1).mainLine z(2).mainLine z(3).mainLine z(4).mainLine], periods{1}, periods{2}, periods{3}, periods{4})
% 
% figure(2)
% legend([y(1).mainLine y(2).mainLine y(3).mainLine y(4).mainLine],  periods{1}, periods{2}, periods{3}, periods{4})
% 
% figure(3)
% legend([x(1).mainLine x(2).mainLine x(3).mainLine x(4).mainLine],  periods{1}, periods{2}, periods{3}, periods{4})