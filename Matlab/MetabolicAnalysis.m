%%%%%%% Metabolic data processing %%%%%%%
%% import data with custom function
clear
%MetData = importMetabolics('D:\DF_uphill_pilot_092619.xlsx')
MetData = importMetabolics('C:\Users\Daniel.Feeney\Dropbox (Boa)\Hike Work Research\Data\bretthike_metabolics.xlsx');
% Find when marker data was not empty
%trials = find(MetData.Marker(4:end) ~= "")
subWt = 85;
% Create a relative minute and second array
MetData.tNew = datetime(MetData.t,'ConvertFrom','excel');
%% Plot if no markers made
plot(MetData.VO2Kg)
hold on
plot(MetData.HR)
hold off
disp('Select start of three regions of interest')
[start, loc] = ginput(3)

figure(1)
plot(MetData.VO2Kg(floor(start(1)):floor(start(1))+20),'Linewidth',2)
hold on
plot(MetData.VO2Kg(floor(start(2)):floor(start(2))+20),'Linewidth',2)
plot(MetData.VO2Kg(floor(start(3)):floor(start(3))+20),'Linewidth',2)
legend('1st Epoch', '2nd Epoch','3rd Epoch')
ylabel('VO2 (ml O2/min/Kg)','FontSize',14)
xlabel('Index','FontSize',14)

figure(2)
plot(MetData.HR(floor(start(1)):floor(start(1))+20),'Linewidth',2)
hold on
plot(MetData.HR(floor(start(2)):floor(start(2))+20),'Linewidth',2)
plot(MetData.HR(floor(start(3)):floor(start(3))+20),'Linewidth',2)
legend('1st Condition', '2nd Condition','3rd Condition')
ylabel('HR','FontSize',14)
xlabel('Index','FontSize',14)

%% Calculate metabolic rate with Brockway equation
firstEpoch = ((16.58 * mean(MetData.VO2(floor(start(1)):floor(start(1))+20))) + (4.51 * mean(MetData.VCO2(floor(start(1)):floor(start(1))+20)))) / subWt
secondEpoch = ((16.58 * mean(MetData.VO2(floor(start(2)):floor(start(2))+20))) + (4.51 * mean(MetData.VCO2(floor(start(2)):floor(start(2))+20)))) / subWt
thirdEpoch = ((16.58 * mean(MetData.VO2(floor(start(3)):floor(start(3))+20))) + (4.51 * mean(MetData.VCO2(floor(start(3)):floor(start(3))+20)))) / subWt

avgHR1 = mean(MetData.HR((floor(start(1)):floor(start(1))+20)))
avgHR2 = mean(MetData.HR((floor(start(2)):floor(start(2))+20)))
avgHR3 = mean(MetData.HR((floor(start(3)):floor(start(3))+20)))

output = [firstEpoch,avgHR1;secondEpoch,avgHR2;thirdEpoch, avgHR3]

%% Plot data if markers were added
% plot(MetData.VO2Kg(trials(3):trials(4))) %LR
% hold on
% plot(MetData.VO2Kg(trials(6):trials(7))) %Standard lace
% plot(MetData.VO2Kg(trials(8):trials(9))) % overlapping panels
% plot(MetData.VO2Kg(trials(10):trials(11))) % Tri Panel
% plot(MetData.VO2Kg(trials(12):trials(13))) % X 
% plot(MetData.VO2Kg(trials(14):trials(15))) % Hierro
% legend('LR','Lace','OP','Tri','X','Hierro')
% 
% %% Calculate metabolic rate with Brockway equation
% LR = ((16.58 * mean(MetData.VO2(trials(3):trials(3)+20))) + (4.51 * mean(MetData.VCO2(trials(3):trials(3)+20)))) / subWt;
% Lace = ((16.58 * mean(MetData.VO2(trials(7) - 20:trials(7)))) + (4.51 * mean(MetData.VCO2(trials(7)-20:trials(7))))) / subWt;
% OP = ((16.58 * mean(MetData.VO2(trials(9) - 20:trials(9)))) + (4.51 * mean(MetData.VCO2(trials(9)-20:trials(9))))) / subWt;
% Tri = ((16.58 * mean(MetData.VO2(trials(11) - 20:trials(11)))) + (4.51 * mean(MetData.VCO2(trials(11)-20:trials(11))))) / subWt;
% X = ((16.58 * mean(MetData.VO2(trials(13) - 20:trials(13)))) + (4.51 * mean(MetData.VCO2(trials(13)-20:trials(13))))) / subWt;
% results.Lace = Lace; results.LR = LR; results.OP = OP; results.Tri = Tri; results.X = X;
% %% Calc percent differences and store 
% pctDiff.LR = ((results.LR - results.Lace) / results.Lace) * 100;
% pctDiff.OP = ((results.OP - results.Lace) / results.Lace) * 100;
% pctDiff.Tri = ((results.Tri - results.Lace) / results.Lace) * 100;
% pctDiff.X = ((results.X - results.Lace) / results.Lace) * 100;