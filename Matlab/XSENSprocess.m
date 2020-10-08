%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Script to process golf IMU data %%%%%%%%%%%%%%%%%%%%%%
%%% notes: This can be used by preprocessing each golf config containing 10
%%% swings, then saving each variable with an appropriate suffix. Angular
%%% velocities are in rad/s, and may be converted by multiplying by 180/Pi.

%% Preprocessing steps. First, extract angles, velocities, etc and save them.
% Find the locations w/ G input and save the angular velocities for each
% config with the correct suffix
% Use genAverages function to generate the angularVelcity_mat for each
% config. Finally, now the time series may be plotted w function
% shadedErrorBar
addpath('C:\Users\Daniel.Feeney\Documents\Trail Run\Run Code')
extractAngle(tree) %Use this line to generate the angular data and rename/save with appropriate suffix (e.g. trial order: angles --> angles_pilot)
Config = 'D'; %This must be changed to allow for auto updating of file names
%% make a plot and save locations
%%%%%%%%% The following lines may be used to partition %%%%%%%%%%%%%
%%%%%%%%%          (by hand) the different swings      %%%%%%%%%%%%%

%%% more automated method to identify swings %%%
findpeaks(angles.R_ankle(3,10:end), 'minpeakdistance', 100, 'MinPeakProminence',20);
[pks,locs] = findpeaks(angles.R_ankle(3,10:end), 'minpeakdistance', 100, 'MinPeakProminence',20);
locs = locs - 75; %-75 steps back 75 frames to what should be approximately the start of the swing. 

%generate averages
genAverageAngles(angles,locs,150,Config);


%% plotting
figure(2)
shadedErrorBar(1:length(angles_D_proc.rAnkle.AB), angles_D_proc.rAnkle.AB, {@mean,@std}, 'lineprops','-b');
