# -*- coding: utf-8 -*-
"""
VO2 Master Unit - multi-trial data processing and visualisation

Uses a SessionDiagnostics.xlsx to segment trials from the VO2 data file.
Trial boundaries:  start = SessionMark timestamp
                   end   = next SessionPause (or SessionComplete for the last
                           trial)

The Details column of each SessionMark is used as the trial label
(e.g. QL_1, SG_2).

Steady-state window: minutes 3-4 after each trial start (configurable).

Run this a cell at a time in Spyder and set the options under "Config" below.
The usage notes that used to sit here described argparse flags that the script
has never had.

Created on Thu Aug 13 11:35:26 2026
@author: max.ferguson
"""
import os
import re
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import messagebox
from TreadmillFunctions import loadProcessingDecisions, saveProcessingDecision


StudyFolder = 'C:/Users/eric.honert/OneDrive - BOA Technology Inc/PFL Team - General/Testing Segments/Outdoor/Hike_Trek/2026_Mechanistic_TrekInField_Scarpa/'

fPath       = StudyFolder + "VO2Master/"
OUTPUT_DIR  = fPath
fileExt     = ".xlsx"
fileExtDiag = "Diagnostics.xlsx"

# The qualitative sheet is the source of truth for configuration names. Taking
# them from here rather than from a list in this file means the next study
# needs no code change, and a SessionMark label that matches nothing in the
# sheet is reported instead of being compiled under whatever it happened to
# say. Correcting a mistyped config in the sheet is then all that is needed.
qualPath = StudyFolder + "CompiledQual_TrekInField.xlsx"

# The 0_ prefix marks this script's own output. Nothing written here is an
# .xlsx today, so none of these listings can pick one up, but the filter
# costs nothing and matches every other script in the repo.
diag = [fName for fName in os.listdir(fPath)
        if fName.endswith(fileExtDiag) and not fName.startswith('0_')]
data = [fName for fName in os.listdir(fPath)
        if fName.endswith(fileExt) and not fName.endswith(fileExtDiag)
        and not fName.startswith('0_')]

save_on = 1
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about subjects with no recorded decision (the normal setting)
# 2 = re-review EVERY subject, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 2

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'VO2MasterField'

# Data-quality decisions recorded by previous runs of this script. Loading
# these is what stops a subject being judged twice. Rejected files are left
# where they are so they can still be opened and looked at - it is this
# record, not the folder, that decides what gets processed.
decisions = loadProcessingDecisions(fPath, scriptName)

# Results from an out-of-date copy of this script will not match anyone
# else's. Asked only when nothing has been decided here yet: a decision file
# means the question was answered on an earlier run of this dataset.
if len(decisions) == 0:
    # Give this dialog its own window and put it in front. It is the first
    # thing the script does, so unlike the data check further down there is no
    # figure on screen to pull focus, and a borrowed default root can open the
    # box behind Spyder - which just looks like the script has hung.
    dialogRoot = tk.Tk()
    dialogRoot.withdraw()
    dialogRoot.attributes('-topmost', True)
    upToDate = messagebox.askyesno("Question", "Is this script up to date with main?",
                                   parent = dialogRoot)
    dialogRoot.destroy()
    # A plain error rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. This stops just as
    # firmly, prints the reason, and leaves the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

# How far before the END of each trial the analysis stops, in seconds.
# Measuring back from the end rather than forward from the start means trials
# of different lengths are compared over the same part of the effort.
WIN_END_BEFORE_END = 15

# The analysis period is N_QUARTERS consecutive blocks of QUARTER_DUR seconds,
# finishing WIN_END_BEFORE_END before the end of the trial. Where it STARTS
# follows from these, so there is no second number to keep in step: say how
# long a block is and how many of them, and the window is whatever that comes
# to. Each block becomes one output row, which is the 'Q' column.
N_QUARTERS = 4
QUARTER_DUR = 30

if N_QUARTERS < 1 or QUARTER_DUR <= 0:
    raise ValueError('N_QUARTERS and QUARTER_DUR must both be positive')

# Built from the blocks, so it cannot fail to be a whole number of them
winWidth = N_QUARTERS * QUARTER_DUR
print('Steady state: ' + str(WIN_END_BEFORE_END + winWidth) + ' s to '
      + str(WIN_END_BEFORE_END) + ' s before the end of each trial, in '
      + str(N_QUARTERS) + ' blocks of ' + str(QUARTER_DUR) + ' s')

REDO_TRIALS = False   # True (or delete the *_trialseg.npy) to re-click a subject

MAKE_PLOTS = False    # True -> also regenerate the per-participant PNGs

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
plt.rc('font',   size=SMALL_SIZE)
plt.rc('axes',   titlesize=SMALL_SIZE)
plt.rc('axes',   labelsize=MEDIUM_SIZE)
plt.rc('xtick',  labelsize=SMALL_SIZE)
plt.rc('ytick',  labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
TRIAL_COLORS = plt.cm.tab10.colors

# ---- Column mapping — CONFIRM these against your actual VO2 file --------
COL_TIME = 'Time[s]'
COL_VO2  = 'VO2[mL/kg/min]'        # absolute VO2 (~2600 range, matches your target)
COL_EE   = 'Calories[kcal/hr]'  # see EE_SCALE note below
COL_VE   = 'Ve[L/min]'
COL_RF   = 'Rf[bpm]'
COL_Press   = 'Pressure[hPa]'
COL_Temp   = 'Temp[C]'
EE_SCALE = 1/60.0               # kcal/hr -> kcal/min (your target EE ~13). Set 1.0 to leave as-is.

def loadSegments(diag_path):
    """
    Parse a SessionDiagnostics file and return trial boundary segments.

    Reads the SessionDiagnostics.xlsx exported by the VO2 Master app and
    extracts the start and end time of each trial based on event types:

      - SessionMark  → trial start; the 'Details' field (e.g. 'QL_1') is used
                       as the trial label.
      - SessionPause → trial end for all trials except the last.
      - SessionComplete → trial end for the final trial if no subsequent
                          SessionPause exists.

    Each SessionMark is paired with the first SessionPause or SessionComplete
    whose timestamp is strictly greater than the mark's timestamp.

    Parameters
    ----------
    diag_path : str
        Absolute or relative path to the SessionDiagnostics .xlsx file.

    Returns
    -------
    list of dict
        One dict per trial, each containing:
          - 'label'   (str)   : trial name from the SessionMark Details column.
          - 'trial'   (int)   : 1-based trial index.
          - 't_start' (float) : trial start time in seconds.
          - 't_end'   (float or None) : trial end time in seconds, or None if
                                        no closing event was found.

    Raises
    ------
    ValueError
        If no SessionMark events are found in the diagnostics file.
    """
    diag = pd.read_excel(diag_path)

    marks    = diag[diag['Type[type]'] == 'SessionMark'].sort_values('Time[s]').reset_index(drop=True)
    pauses   = diag[diag['Type[type]'] == 'SessionPause'].sort_values('Time[s]')['Time[s]'].tolist()
    complete = diag[diag['Type[type]'] == 'SessionComplete']['Time[s]']
    end_times = pauses + ([float(complete.iloc[0])] if len(complete) > 0 else [])

    if len(marks) == 0:
        raise ValueError('No SessionMark events found in diagnostics file.')

    segments = []
    for i, row in marks.iterrows():
        t_start = float(row['Time[s]'])
        label   = str(row['Details']).strip() if pd.notna(row['Details']) else f'Trial_{i+1}'
        # find the first end-time that comes after this mark
        t_end_candidates = [t for t in end_times if t > t_start]
        t_end = t_end_candidates[0] if t_end_candidates else None
        segments.append({'label': label, 'trial': i + 1,
                         't_start': t_start, 't_end': t_end})

    return segments

def plotTimeseriesAllTrials(trial_data):
    """
    Plot a 4-panel time-series with all trials overlaid in separate colours.

    Each panel shows one metabolic or ventilatory metric on the y-axis and
    time relative to trial start (in minutes) on the x-axis. Each trial's own
    steady-state collection window ('win_start'/'win_end', absolute seconds)
    is converted to trial-relative time and shaded in that trial's colour, so
    you can see at a glance whether the windows line up across trials.

    Panels: VO2 (mL/kg/min), Minute Ventilation (L/min),
            Energy Expenditure (kcal/hr), Breathing Rate (bpm).

    Parameters
    ----------
    trial_data : list of dict
        List produced by the main slicing loop. Each dict must contain:
          - 'df'        (pd.DataFrame) : trial data with 't_rel' column.
          - 'label'      (str)         : trial name for the legend.
          - 't_start'    (float)       : trial start time, absolute seconds.
          - 'win_start'  (float)       : steady-state window start, absolute seconds.
          - 'win_end'    (float)       : steady-state window end, absolute seconds.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object (not yet saved or displayed).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('VO2 Master — All Trials Overlaid', fontsize=BIGGER_SIZE, fontweight='bold')

    panels = [
        ('VO2[mL/kg/min]',    'VO₂',                    'mL/kg/min', axes[0, 0]),
        ('Ve[L/min]',          'Minute Ventilation (VE)', 'L/min',     axes[0, 1]),
        ('Calories[kcal/hr]',  'Energy Expenditure',      'kcal/hr',   axes[1, 0]),
        ('Rf[bpm]',            'Breathing Rate (Rf)',      'bpm',       axes[1, 1]),
    ]

    for col, title, unit, ax in panels:
        for i, td in enumerate(trial_data):
            df    = td['df']
            lbl   = td['label']
            clr   = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            t_min = df['t_rel'] / 60
            ax.plot(t_min, df[col], color=clr, linewidth=1.5, label=lbl)

            win_start_rel = (td['win_start'] - td['t_start']) / 60
            win_end_rel   = (td['win_end']   - td['t_start']) / 60
            ax.axvspan(win_start_rel, win_end_rel, alpha=0.15, color=clr)
            ax.axvline(win_start_rel, color=clr, linewidth=0.8, linestyle='--')
            ax.axvline(win_end_rel,   color=clr, linewidth=0.8, linestyle='--')

        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel('Time in trial (min)')
        ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def plotSSBarComparison(df_sub):
    """
    One bar group per metric, one bar per trial, drawn from the compiled rows.

    Bar heights are the mean of that trial's quarter values - the same numbers
    written to the output file. This takes the compiled records rather than the
    trial data on purpose. It used to recompute its own steady-state mean over
    a single window instead of the quarters, inclusive at both ends instead of
    half-open, over relative time instead of absolute, and without the EE
    scaling, so its EE bars read 60x the EE column in the file and changing the
    window moved the file but not the figure.

    Parameters
    ----------
    df_sub : pd.DataFrame
        The compiled rows for one subject, as built in the trial loop. Needs
        the Config, Order and metric columns.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (not saved or displayed).

    """
    metrics = [('VO2', 'VO2\n(mL/kg/min)'),
               ('Ve',  'VE\n(L/min)'),
               ('Rf',  'Rf\n(bpm)'),
               ('EE',  'EE\n(kcal/min)')]

    if len(df_sub) == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title('No compiled rows to plot')
        return fig

    # One row per trial: the mean across that trial's quarters
    perTrial = df_sub.groupby(['Config', 'Order'], as_index=False).mean(numeric_only=True)
    perTrial = perTrial.sort_values('Order').reset_index(drop=True)

    nMetrics = len(metrics)
    nTrials = len(perTrial)
    x = np.arange(nMetrics)
    width = 0.8 / nTrials

    fig, ax = plt.subplots(figsize=(10, 5))
    for ii in range(nTrials):
        row = perTrial.iloc[ii]
        label = str(row['Config']) + '_' + str(int(row['Order']))
        clr = TRIAL_COLORS[ii % len(TRIAL_COLORS)]

        vals = []
        for col, _ in metrics:
            if col in perTrial.columns:
                vals.append(row[col])
            else:
                vals.append(np.nan)

        offset = (ii - nTrials / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=clr, edgecolor='white')

        # Annotation headroom from this trial's own bars, worked out once
        # rather than inside the labelling loop
        finite = [v for v in vals if not np.isnan(v)]
        if len(finite) == 0:
            continue
        headroom = max(finite) * 0.01

        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            xMid = bar.get_x() + bar.get_width() / 2
            yTop = bar.get_height() + headroom
            ax.text(xMid, yTop, '%.1f' % val, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_title('Steady state by trial, averaged over the compiled quarters',
                 fontsize=MEDIUM_SIZE, fontweight='bold')
    ax.set_ylabel('Mean of the steady-state quarters')
    ax.legend(fontsize=11)
    plt.tight_layout()
    return fig

def plotEnvironment(trial_data):
    """
    Plot ambient environmental conditions recorded by the VO2 Master unit.

    Produces a 1x3 panel figure showing barometric pressure, temperature, and
    relative humidity over time for each trial. Trials are overlaid on the
    same axes in matching colours to the other plots, and each trial's own
    steady-state window is shaded in that trial's colour for reference.

    Parameters
    ----------
    trial_data : list of dict
        List produced by the main slicing loop. Each dict must contain:
          - 'df'        (pd.DataFrame) : trial data with 't_rel' column.
          - 'label'      (str)         : trial name for the legend.
          - 't_start'    (float)       : trial start time, absolute seconds.
          - 'win_start'  (float)       : steady-state window start, absolute seconds.
          - 'win_end'    (float)       : steady-state window end, absolute seconds.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object (not yet saved or displayed).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('VO2 Master — Environmental Conditions', fontsize=BIGGER_SIZE, fontweight='bold')

    env_panels = [
        ('Pressure[hPa]', 'Pressure',    'hPa', axes[0]),
        ('Temp[C]',        'Temperature', '°C',  axes[1]),
        ('HUM[%RH]',       'Humidity',    '%RH', axes[2]),
    ]

    for col, title, unit, ax in env_panels:
        for i, td in enumerate(trial_data):
            df  = td['df']
            lbl = td['label']
            clr = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            ax.plot(df['t_rel'] / 60, df[col], color=clr, linewidth=1.5, label=lbl)
            win_start_rel = (td['win_start'] - td['t_start']) / 60
            win_end_rel   = (td['win_end']   - td['t_start']) / 60
            ax.axvspan(win_start_rel, win_end_rel, alpha=0.15, color=clr)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel('Time in trial (min)')
        ax.legend(fontsize=10)

    plt.tight_layout()
    return fig

def subjectKey(fname):
    """
    Reduce a VO2 Master file name to the participant name.

    A participant contributes more than one file - the data export and the
    session diagnostics - and the two have to be paired. Stripping the part
    that says which kind of file it is leaves the participant name, so both
    files produce the same key and can be matched in a dictionary.

        BenWells.xlsx                     ->  BenWells
        BenWells_SessionDiagnostics.xlsx  ->  BenWells

    The suffixes are tried longest first and only one is removed, so
    "SessionDiagnostics.xlsx" is matched before the ".xlsx" inside it. The
    trailing separator is then stripped, which is what lets the same function
    read both the current naming and the earlier files that ran the name
    straight into the suffix (BenWellsSessionDiagnostics.xlsx).

    Matching on an exact key rather than on a prefix matters: a prefix test
    would pair a participant called Ben with BenWells_SessionDiagnostics.xlsx.

    Parameters
    ----------
    fname : str
        File name, with no directory part, e.g. "BenWells.xlsx" or
        "BenWells_SessionDiagnostics.xlsx"

    Returns
    -------
    str
        The participant name. A name that matches none of the known suffixes
        is returned with only its separators trimmed, so an unexpected file
        becomes its own key rather than silently joining another participant.

    """
    key = fname
    for suf in ('SessionDiagnostics.xlsx', 'Diagnostics.xlsx', '.xlsx'):
        if key.endswith(suf):
            key = key[:-len(suf)]
            break
    return key.rstrip(' _-')

diag_by_subject = {subjectKey(d): d for d in diag}

# Configuration names, as the qual sheet spells them. Keyed on the lower-cased
# name so a label does not have to match the sheet's capitalisation.
qual = pd.read_excel(qualPath)
knownConfigs = sorted({str(cfg).strip() for cfg in qual['Config'].dropna()})
configByName = {}
for cfg in knownConfigs:
    configByName[cfg.lower()] = cfg
print('Configurations in the qual sheet: ' + ', '.join(knownConfigs))

# Labels that matched no configuration, collected rather than printed as they
# happen so they can be read as one list at the end of the run
disregardedLabels = {}

# Steady-state blocks that had no breaths in them at all, so no row was
# written. Keyed on the trial, because what matters is which trial came out
# short rather than how many blocks were lost overall.
missingBlocks = {}

# Blocks that had breaths but whose metric still averaged to NaN, which is a
# different failure: the row exists, the number in it does not
nanBlocks = {}

all_results = []

for vo2_name in sorted(data):
    subject = subjectKey(vo2_name)
    diag_name = diag_by_subject.get(subject)

    if diag_name is None:
        print('  ! no diagnostics file for ' + subject + ', skipping')
        continue

    # A subject rejected in a previous run is skipped. The files are still
    # there to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(vo2_name, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + subject)
        continue

    # A subject already accepted is processed without being asked about again
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + subject)
        checkThisSubject = 0
    else:
        checkThisSubject = data_check

    print('Processing ' + subject + ' ...')

    #______________________________________________________________________
    # Load this subject's files. These stay in the namespace rather than
    # living inside a per-subject function, so a breakpoint anywhere below
    # has df_vo2, segments and trial_data to look at.
    vo2_path = os.path.join(fPath, vo2_name)
    diag_path = os.path.join(fPath, diag_name)
    filepath = os.path.dirname(vo2_path)

    df_vo2 = pd.read_excel(vo2_path, header=0)
    df_vo2 = df_vo2.sort_values('Time[s]').reset_index(drop=True)

    segments = loadSegments(diag_path)

    #______________________________________________________________________
    # Keep only the marks whose configuration is named in the qual sheet, and
    # work out the configuration and the order here rather than at the end.
    # Warmups, aborted trials and mistyped configs are dropped before any
    # clicking or plotting happens, so nothing is segmented or shown that is
    # not going to be compiled.
    #
    # The label carries both: BOA_2 is configuration BOA, order 2. Order used
    # to be the sequential position of the mark, which is off by one for
    # every subject whose first mark is a warmup.
    trialSegments = []
    for seg in segments:
        parts = re.split(r'[_-]', str(seg['label']))
        configToken = parts[0].strip().lower()

        if configToken not in configByName:
            if str(seg['label']) not in disregardedLabels:
                disregardedLabels[str(seg['label'])] = []
            disregardedLabels[str(seg['label'])].append(subject)
            continue

        seg['config'] = configByName[configToken]

        if len(parts) > 1 and parts[1].strip().isdigit():
            seg['order'] = int(parts[1].strip())
        else:
            # The configuration matched but the label carries no number, so
            # fall back to where the mark sat in the file and say so
            seg['order'] = seg['trial']
            print('   no order in label ' + str(seg['label'])
                  + ', using mark position ' + str(seg['order']))

        trialSegments.append(seg)

    if len(trialSegments) == 0:
        print('   no marks match a configuration in the qual sheet, skipping '
              + subject)
        continue

    #______________________________________________________________________
    # Cut the recording into trials. The start and end of each trial are
    # clicked once and cached to <subject>_trialseg.npy; later runs load the
    # clicks and skip the prompt. Delete the .npy or set REDO_TRIALS to
    # re-select.
    cache_path = os.path.join(filepath, subject + '_trialseg.npy')
    if os.path.exists(cache_path) and not REDO_TRIALS:
        clicks = np.load(cache_path, allow_pickle=True).item()
    else:
        clicks = {}

    trial_data = []
    updated = False
    for seg in trialSegments:
        key = str(seg['trial']).zfill(2) + '_' + str(seg['label'])

        if key in clicks:
            # Already clicked on an earlier run
            t0, t_end = clicks[key]
        else:
            print('Select start and end for: ' + str(seg['label'])
                  + ' (trial ' + str(seg['trial']) + ')')
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df_vo2[COL_TIME], df_vo2[COL_VO2], label='VO2')
            ax.set_title('Trial ' + str(seg['trial']) + ' - ' + str(seg['label'])
                         + '\nSelect START then END')
            ax.set_xlabel('Time')
            ax.set_ylabel('VO2')
            if seg['t_start'] is not None:
                ax.axvline(seg['t_start'], color='g', ls='--', alpha=0.5, label='Diag start')
            if seg['t_end'] is not None:
                ax.axvline(seg['t_end'], color='r', ls='--', alpha=0.5, label='Diag end')
            ax.legend()
            plt.tight_layout()

            pts = np.asarray(plt.ginput(2, timeout=120))
            plt.close(fig)
            t0 = float(pts[0, 0])
            t_end = float(pts[1, 0])
            clicks[key] = (t0, t_end)
            updated = True

        mask = (df_vo2[COL_TIME] >= t0) & (df_vo2[COL_TIME] < t_end)
        tdf = df_vo2[mask].copy()
        tdf['t_rel'] = tdf[COL_TIME] - t0

        trial_data.append({'label': seg['label'], 'trial': seg['trial'],
                           'config': seg['config'], 'order': seg['order'],
                           't_start': t0, 't_end': t_end, 'df': tdf,
                           'win_start': t_end - WIN_END_BEFORE_END - winWidth,
                           'win_end': t_end - WIN_END_BEFORE_END})

    # Only written when something new was clicked
    if updated:
        np.save(cache_path, np.array(clicks, dtype=object))

    #______________________________________________________________________
    # Data check: every trial for this subject on one figure, asked about
    # once. This used to step through a figure per trial, advanced by
    # plt.waitforbuttonpress(), which only registers a click on the figure
    # itself - a click anywhere else waits forever.
    answer = True # if data check is off.
    if checkThisSubject >= 1:
        nTrials = len(trial_data)
        figHeight = 2.4 * nTrials
        fig, axes = plt.subplots(nTrials, 1, figsize=(12, figHeight))
        # subplots returns a bare Axes when there is only one trial
        axes = np.atleast_1d(axes)

        for jj in range(nTrials):
            td = trial_data[jj]
            ax = axes[jj]
            ax.plot(td['df'][COL_TIME], td['df'][COL_VO2], label='VO2')
            ax.axvline(td['t_start'], color='g', linestyle='--', label='Trial start')
            if td['t_end'] is not None:
                ax.axvline(td['t_end'], color='r', linestyle='--', label='Trial end')

            ax.axvspan(td['win_start'], td['win_end'], alpha=0.2, color='orange',
                       label='Steady-state window')

            ax.set_ylabel(str(td['label']))

        axes[0].set_title(subject + ' - VO2 by trial, shaded = steady-state window')
        axes[0].legend(fontsize=9)
        axes[nTrials-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.show(block=False)
        # Let the figure draw before the modal dialog takes the event loop,
        # otherwise the window can come up blank behind the question
        plt.pause(0.1)

        # The dialog gets its own window, put in front. Called bare it borrows
        # the default root - and the root made for the up-to-date question at
        # startup has already been destroyed - which opens the box behind
        # Spyder, where it looks like nothing happened at all.
        dialogRoot = tk.Tk()
        dialogRoot.withdraw()
        dialogRoot.attributes('-topmost', True)
        answer = messagebox.askyesno("Data check",
                                     'Is VO2 data clean for ' + subject + '?',
                                     parent = dialogRoot)
        dialogRoot.destroy()
        plt.close(fig)

    # Record the decision straight away, inside the loop, so a run that fails
    # later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, vo2_name, 'bad',
                                   Subject=subject,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, vo2_name, 'good',
                                   Subject=subject)

    if answer == False:
        print('Excluding ' + subject)
        continue

    #______________________________________________________________________
    # Summarise each quarter of the steady-state window. The configuration
    # and the order were resolved before the segmentation, so they are read
    # here rather than parsed out of the label a second time.
    records = []
    for td in trial_data:
        config = td['config']
        order = td['order']

        for ii in range(1, N_QUARTERS + 1):
            winStart = td['win_start'] + QUARTER_DUR*ii - QUARTER_DUR
            winEnd = td['win_start'] + QUARTER_DUR*ii

            inWindow = (td['df'][COL_TIME] >= winStart) & (td['df'][COL_TIME] < winEnd)
            w = td['df'][inWindow]
            trialKey = (subject, config, order)

            if w.empty:
                # No breaths in this block, so no row. Said out loud: nothing
                # downstream can tell a trial that lost a block from a trial
                # that was only ever three blocks long.
                if trialKey not in missingBlocks:
                    missingBlocks[trialKey] = []
                missingBlocks[trialKey].append(ii)
                print('   ' + subject + ' ' + config + '_' + str(order)
                      + ': block ' + str(ii) + ' of ' + str(N_QUARTERS)
                      + ' has no breaths, no row written')
                continue

            # Every mean below skips NaN, which pandas does by default. That
            # is doing real work here: VO2 and Calories are NaN on roughly one
            # breath in twelve, so taking a plain average of the raw values
            # would return NaN for most blocks.
            record = {'Subject': subject, 'Config': config, 'Order': order, 'Q': ii}
            record['EE'] = w[COL_EE].mean() * EE_SCALE if COL_EE in w else np.nan
            record['VO2'] = w[COL_VO2].mean() if COL_VO2 in w else np.nan
            record['Ve'] = w[COL_VE].mean() if COL_VE in w else np.nan
            record['Rf'] = w[COL_RF].mean() if COL_RF in w else np.nan
            record['Pressure'] = w[COL_Press].mean() if COL_Press in w else np.nan
            record['Temp'] = w[COL_Temp].mean() if COL_Temp in w else np.nan
            records.append(record)

            # A block can hold breaths and still average to NaN, when every
            # breath in it failed for that metric. The row is kept - the block
            # did happen - but the gap is collected rather than left to be
            # found in the output file.
            nanMetrics = []
            for metricName in ('EE', 'VO2', 'Ve', 'Rf', 'Pressure', 'Temp'):
                if np.isnan(record[metricName]):
                    nanMetrics.append(metricName)

            if len(nanMetrics) > 0:
                if trialKey not in nanBlocks:
                    nanBlocks[trialKey] = []
                nanBlocks[trialKey].append((ii, nanMetrics))

    df_sub = pd.DataFrame(records)
    all_results.append(df_sub)

    if MAKE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.join(OUTPUT_DIR, "Plots", subject)
        plotTimeseriesAllTrials(trial_data).savefig(base + '_timeseries.png',  dpi=150, bbox_inches='tight')
        plotSSBarComparison(df_sub).savefig(base + '_ss_compare.png',  dpi=150, bbox_inches='tight')
        plotEnvironment(trial_data).savefig(base + '_environment.png', dpi=150, bbox_inches='tight')
        plt.close('all')

#______________________________________________________________________________
# Anything the qual sheet did not account for. Warmups and aborted trials are
# expected here; a config name that is only one of these away from a real one
# is a typo, in the sheet or in the SessionMark, and fixing the sheet is
# enough to bring those trials back in.
print('')
print('Labels with no matching configuration in the qual sheet:')
if len(disregardedLabels) == 0:
    print('   none')
else:
    for label in sorted(disregardedLabels):
        subjects = disregardedLabels[label]
        print('   %-26s %2d trial(s): %s'
              % (label, len(subjects), ', '.join(sorted(set(subjects)))))
print('')

# A trial that is short of blocks contributes fewer rows than the others, and
# weights differently in anything that averages across trials.
print('Trials short of a steady-state block (no breaths in that block):')
if len(missingBlocks) == 0:
    print('   none')
else:
    for trialKey in sorted(missingBlocks):
        blocks = missingBlocks[trialKey]
        blockList = ', '.join(str(b) for b in blocks)
        print('   %-24s %-6s order %d   missing block(s) %s of %d'
              % (trialKey[0], trialKey[1], trialKey[2], blockList, N_QUARTERS))
print('')

# These rows exist but carry NaN for the metrics listed, because every breath
# in that block failed for them
print('Blocks with breaths but no usable value:')
if len(nanBlocks) == 0:
    print('   none')
else:
    for trialKey in sorted(nanBlocks):
        for blockNo, metricNames in nanBlocks[trialKey]:
            print('   %-24s %-6s order %d   block %d: no value for %s'
                  % (trialKey[0], trialKey[1], trialKey[2], blockNo,
                     ', '.join(metricNames)))
print('')

compiled = pd.concat(all_results, ignore_index=True)
compiled = compiled[['Subject', 'Config', 'Order', 'Q', 'EE', 'VO2', 'Ve', 'Rf',
                     'Pressure', 'Temp']]

out_csv = os.path.join(OUTPUT_DIR, '0_CompiledMetabolics.csv')
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    compiled.to_csv(out_csv, index=False)



