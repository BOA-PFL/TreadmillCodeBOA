clear
clc

input_dir = 'C:\Users\kate.harrison\Dropbox (Boa)\EndurancePerformance\NewBalanceRC_May2021\Metabolics';

cd(input_dir)
files = dir('*.xlsx');
dataList = {files.name};
[f,~] = listdlg('PromptString','Select data files','SelectionMode','multiple','ListString',dataList);

NumbTrials = length(f);

Subject = cell(2, 1);
Shoe = cell(2, 1);

VO2 = zeros(2,1);
EE = zeros(2,1);

r = 0;

for i = 1:NumbTrials
    
    FileName = dataList(f(i));
    FileLoc = char(strcat(input_dir,'\', FileName));
    vo2 = xlsread(FileLoc, 'Data', 'O:O');
    vo2 = vo2/60;
    vco2 = xlsread(FileLoc, 'Data', 'P:P');
    vco2 = vco2/60;
    m = xlsread(FileLoc, 'Data', 'B7');
    names = split(FileName, ["_"," ","."]);
    sub = names{1};
    shoe = names{3};

    energy = (16.58*vo2 + 4.51*vco2)/m;
    plot(energy)
    
    [start,~] = ginput(1); % find start of steady state
    start = round(start); 
    
    [stop,~] = ginput(1); % end of run/steady state
    stop = round (stop);
    
    l = (stop - start)/6; %length of epochs after steady state
    l = floor(l);
    
    
    for j = 1:6
        
        s = start + ((j-1)*l);
        Subject{r+j} = sub;
        Shoe{r+j} = shoe;
        EE(r+j) = mean(energy(s:s+l));
        VO2(r+j) = mean(vo2(s:s+l));       
    end 
    
    r = length(EE);
end

disp('done!')

Titles = {'SubjectName', 'Shoe', 'EE', 'VO2'};
EE = num2cell(EE);
VO2 = num2cell(VO2);
data = horzcat(Subject, Shoe, EE, VO2);
data = vertcat(Titles, data);

writecell(data, 'CompiledMetabolicData.csv')