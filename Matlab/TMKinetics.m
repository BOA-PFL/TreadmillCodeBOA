%% TM with kinetics 
clear
subBW = 85;
addpath('C:\Users\Daniel.Feeney\Documents\Trail Run\Run Code')
%[file, path] = uigetfile;
%kinData = importTMKinetics(strcat(path,file));
kinData = importTMKinetics('C:\Users\Daniel.Feeney\Dropbox (Boa)\Hike Work Research\Data\TMHike\brett_up_SD - Kinetics.txt');
forceThresh = 50;

%fnameLong = strsplit(file,'_'); condLong = strsplit(fnameLong{3},'-');
%subj = fnameLong{1}; cond = condLong{1}(1:2); %splitting the file name to obtain subject and condition
%% Kinetics and zero'ing all kinetics when left foot is not on belt
% first, remove low Z force values
%filt_forceZ = filt_forceZ-35;
filt_forceZ = kinData.LeftZForce * -1;
filt_forceZ(filt_forceZ<forceThresh) = 0; %detrend the Z signals so they start at 0

% delimt when there is force on the left side
leftStrike = zeros(length(filt_forceZ),1);
leftStrike(find(filt_forceZ)) = 1;
% element-wise multiple binary leftStrike vector with each joint
% Powers
LAnklePower = leftStrike .* kinData.LAnklePower;
LKneePower = leftStrike .* kinData.LKneePower;
LHipPower = leftStrike .* kinData.LHipPower;
% Moments
LAnkleMomX = leftStrike .* kinData.LAnkleMomentx; LAnkleMomY = leftStrike .* kinData.LAnkleMomenty; LAnkleMomZ = leftStrike .* kinData.LAnkleMomentz;
LKneeMomX = leftStrike .* kinData.LKneeMomentX; LKneeMomY = leftStrike .* kinData.LKneeMomentY; LKneeMomZ = leftStrike .* kinData.LKneeMomentZ;
LHipMomX = leftStrike .* kinData.LHipMomentx; LHipMomY = leftStrike .* kinData.LHipMomenty; LHipMomZ = leftStrike .* kinData.LHipMomentz;

%% delimit initial contact and toe off
ic = zeros(60,1);
count = 1;
for step = 1:length(leftStrike)-1
    if (filt_forceZ(step) == 0 & filt_forceZ(step + 1) > 30)
        ic(count) = step;
        count = count + 1;
    end
end

to = zeros(60,1);
count = 1;
for step = 1:length(leftStrike)-1
    if (filt_forceZ(step) > 30 & filt_forceZ(step + 1) == 0)
        to(count) = step;
        count = count + 1;
    end
end
% trim first contact/toe off if not a full step
if (ic(1) > to(1))
   to(1) = []; 
end
ic(ic == 0) = []; to(to == 0) = [];

if (ic(end) > to(end))
   ic(end) = []; 
end
% Calculate step lengths
stepLens = to - ic;

%% extract features from each step length

for step = 1:length(stepLens)
    %powers
    maxAnklePow(step) = max(LAnklePower(ic(step):to(step)));
    maxKneePow(step) = max(LKneePower(ic(step):to(step)));
    maxHipPow(step) = max(LHipPower(ic(step):to(step)));
    %work
    AnkleW(step) = sum(LAnklePower(ic(step):to(step)));
    KneeW(step) = sum(LKneePower(ic(step):to(step)));
    HipW(step) = sum(LHipPower(ic(step):to(step)));
    %Moments
    AnkleMx(step) = min(LAnkleMomX(ic(step):to(step))); AnkleMy(step) = max(LAnkleMomY(ic(step):to(step))); AnkleMz(step) = min(LAnkleMomZ(ic(step):to(step)));
    KneeMx(step) = max(LKneeMomX(ic(step):to(step))); KneeMy(step) = max(LKneeMomY(ic(step):to(step))); KneeMz(step) = max(LKneeMomZ(ic(step):to(step)));
    HipMx(step) = max(LHipMomX(ic(step):to(step))); HipMy(step) = max(LHipMomY(ic(step):to(step))); HipMz(step) = max(LHipMomZ(ic(step):to(step)));
end

output = [stepLens'; maxAnklePow; maxKneePow; maxHipPow; AnkleW; KneeW; HipW; AnkleMx; AnkleMy; AnkleMz; KneeMx; KneeMy; KneeMz; HipMx; HipMy; HipMz]';

%% plots to visualize.
figure
plot(filt_forceZ)
hold on
plot(LAnklePower)
plot(LKneePower)
plot(LHipPower)
ylabel('Joint Power (W)')
title('Joint Powers')
legend('Force','Ankle','Knee','Hip')
xlim([200 600])

%% Ankle moments
figure
yyaxis left
plot(LAnkleMomX)
hold on
plot(LAnkleMomY)
plot(LAnkleMomZ)
yyaxis right
plot(filt_forceZ)
title('Ankle Moments')
ylabel('Joint Moment (Nm)')
legend('X','Y','Z','Force')
xlim([200 600])
%% Knee moments
figure
yyaxis left
plot(LKneeMomX)
hold on
plot(LKneeMomY)
plot(LKneeMomZ)
yyaxis right
plot(filt_forceZ)
title('Knee Moments')
ylabel('Joint Moment (Nm)')
legend('X','Y','Z','Force')
xlim([200 600])

%% Hip moments
yyaxis left
plot(LHipMomX)
hold on
plot(LHipMomY)
plot(LHipMomZ)
yyaxis right
plot(filt_forceZ)
title('Hip Moments')
ylabel('Joint Moment (Nm)')
legend('X','Y','Z','Force')
xlim([200 600])





