%% run plotting data
clear
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesAJesse.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesBJesse.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesCJesse.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesDJesse.mat')

load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesARus.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesBRuss.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesCRuss.mat')
load('C:\Users\Daniel.Feeney\Dropbox (Boa)\Vongo\Vongo_xsensData\Processed\AnglesDRuss.mat')

%% make plots
figure(1)
h(1) = shadedErrorBar(1:length(angles_A_proc.rAnkle.AB), angles_A_proc.rAnkle.AB, {@mean,@std}, 'lineprops','-r');
title('Ankle Inversion/Eversion')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rAnkle.AB), angles_B_proc.rAnkle.AB + 5, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rAnkle.AB), angles_C_proc.rAnkle.AB - 5, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rAnkle.AB), angles_D_proc.rAnkle.AB, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine], 'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')
xlim([0 70])

figure(2)
h(1) = shadedErrorBar(1:length(angles_A_proc.rAnkle.DORSI), angles_A_proc.rAnkle.DORSI, {@mean,@std}, 'lineprops','-r');
title('Ankle Dorsi/plantar flexion')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rAnkle.DORSI), angles_B_proc.rAnkle.DORSI, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rAnkle.DORSI), angles_C_proc.rAnkle.DORSI, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rAnkle.DORSI), angles_D_proc.rAnkle.DORSI, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine],  'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')
xlim([0 70])

figure(3)
h(1) = shadedErrorBar(1:length(angles_A_proc.rKnee.INT), angles_A_proc.rKnee.INT, {@mean,@std}, 'lineprops','-r');
title('Knee internal/external rotation')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rKnee.INT), angles_B_proc.rKnee.INT, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rKnee.INT), angles_C_proc.rKnee.INT, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rKnee.INT), angles_D_proc.rKnee.INT, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine],  'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')
xlim([0 70])

figure(4)
h(1) = shadedErrorBar(1:length(angles_A_proc.rKnee.AB), angles_A_proc.rKnee.AB, {@mean,@std}, 'lineprops','-r');
title('Knee Ab/Adduction')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rKnee.AB), angles_B_proc.rKnee.AB, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rKnee.AB), angles_C_proc.rKnee.AB, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rKnee.AB), angles_D_proc.rKnee.AB, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine],  'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')
xlim([0 70])

figure(5)
h(1) = shadedErrorBar(1:length(angles_A_proc.rHip.INT), angles_A_proc.rHip.INT, {@mean,@std}, 'lineprops','-r');
title('Hip internal/external rotation')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rHip.INT), angles_B_proc.rHip.INT, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rHip.INT), angles_C_proc.rHip.INT, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rHip.INT), angles_D_proc.rHip.INT, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine],  'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')


figure(6)
h(1) = shadedErrorBar(1:length(angles_A_proc.rHip.AB), angles_A_proc.rHip.AB, {@mean,@std}, 'lineprops','-r');
title('Hip Ab/Adduction')
hold on
h(2) = shadedErrorBar(1:length(angles_B_proc.rHip.AB), angles_B_proc.rHip.AB, {@mean,@std}, 'lineprops','-b');
h(3) = shadedErrorBar(1:length(angles_C_proc.rHip.AB), angles_C_proc.lHip.AB, {@mean,@std}, 'lineprops','-g');
h(4) = shadedErrorBar(1:length(angles_D_proc.rHip.AB), angles_D_proc.rHip.AB, {@mean,@std}, 'lineprops','-k');
legend([h(1).mainLine h(2).mainLine h(3).mainLine h(4).mainLine],  'A','B','C','D')
ylabel('Angle')
xlabel('Time (10 ms)')
xlim([0 80])
%% calculating maxes
[max(angles_A_proc.rAnkle.AB,[],2), max((-1 .* angles_A_proc.rAnkle.AB),[],2), max(angles_A_proc.rKnee.AB,[],2),...
    max((-1 .* angles_A_proc.rKnee.AB),[],2), max(angles_A_proc.lAnkle.AB,[],2), max((-1 .* angles_A_proc.lAnkle.AB),[],2)]

[max(angles_B_proc.rAnkle.AB,[],2), max((-1 .* angles_B_proc.rAnkle.AB),[],2), max(angles_B_proc.rKnee.AB,[],2),...
    max((-1 .* angles_B_proc.rKnee.AB),[],2), max(angles_B_proc.lAnkle.AB,[],2), max((-1 .* angles_B_proc.lAnkle.AB),[],2)]

[max(angles_C_proc.rAnkle.AB,[],2), max((-1 .* angles_C_proc.rAnkle.AB),[],2), max(angles_C_proc.rKnee.AB,[],2),...
    max((-1 .* angles_C_proc.rKnee.AB),[],2), max(angles_C_proc.lAnkle.AB,[],2), max((-1 .* angles_C_proc.lAnkle.AB),[],2)]

[max(angles_D_proc.rAnkle.AB,[],2), max((-1 .* angles_D_proc.rAnkle.AB),[],2), max(angles_D_proc.rKnee.AB,[],2),...
    max((-1 .* angles_D_proc.rKnee.AB),[],2), max(angles_D_proc.lAnkle.AB,[],2), max((-1 .* angles_D_proc.lAnkle.AB),[],2)]