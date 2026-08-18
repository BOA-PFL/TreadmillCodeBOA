# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:35:26 2026

@author: max.ferguson
"""

# -*- coding: utf-8 -*-
"""
VO2 Master Unit — Multi-Trial Data Processing & Visualization
-------------------------------------------------------------
Uses a SessionDiagnostics.xlsx to segment trials from the VO2 data file.
Trial boundaries:  start = SessionMark timestamp
                   end   = next SessionPause (or SessionComplete for the last trial)
 
The Details column of each SessionMark is used as the trial label (e.g. QL_1, SG_2).
 
Steady-state window: minutes 3-4 after each trial start (configurable).
 
Usage
-----
  python vo2master_plot.py
  python vo2master_plot.py --vo2  MyData.xlsx --diag MySessionDiagnostics.xlsx
  python vo2master_plot.py --ss_start 120 --ss_end 180
  python vo2master_plot.py --no_save   # display interactively
"""
import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tkinter import messagebox
 
 # ---------------------------------------------------------------------------
 # Data loading
 # ---------------------------------------------------------------------------
def load_vo2(filepath: str) -> pd.DataFrame:
     """
     Load a VO2 Master unit data file and return a sorted DataFrame.
 
     Reads the Excel export from the VO2 Master app, which contains one row
     per breath with an absolute timestamp in the 'Time[s]' column. Rows are
     sorted ascending by time so that downstream slicing is safe regardless of
     the order they appear in the file.
 
     Parameters
     ----------
     filepath : str
         Absolute or relative path to the VO2 Master .xlsx data file.
 
     Returns
     -------
     pd.DataFrame
         DataFrame with all original columns, sorted by 'Time[s]'.
     """
     df = pd.read_excel(filepath, header=0)
     df = df.sort_values('Time[s]').reset_index(drop=True)
     return df
 
 
def load_segments(diag_path: str) -> list:
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
 
 
 # ---------------------------------------------------------------------------
 # Analysis helpers
 # --------------------------------------------------------------------------- 
def ss_mean(df: pd.DataFrame, t_col: str, metric: str,
             t_start: float, t_end: float) -> float:
     """
     Compute the mean of a metric over a steady-state time window.
 
     Selects all rows where the time column falls within [t_start, t_end]
     (inclusive) and returns the mean of the specified metric column, ignoring
     NaN values. Returns NaN if no valid data points exist in the window.
 
     Parameters
     ----------
     df : pd.DataFrame
         Trial DataFrame containing at least the time column and the metric.
     t_col : str
         Name of the time column (e.g. 't_rel' for relative trial time in seconds).
     metric : str
         Name of the column to average (e.g. 'VO2[mL/kg/min]').
     t_start : float
         Window start time in seconds (inclusive).
     t_end : float
         Window end time in seconds (inclusive).
 
     Returns
     -------
     float
         Mean value within the window, or np.nan if the window is empty.
     """
     mask = (df[t_col] >= t_start) & (df[t_col] <= t_end)
     vals = df.loc[mask, metric].dropna()
     return float(np.mean(vals)) if len(vals) > 0 else np.nan
 
 
def compute_trial_summary(trial_df: pd.DataFrame, ss_start: float, ss_end: float) -> dict:
     """
     Compute steady-state means for all key metabolic and ventilatory metrics.
 
     Calls ss_mean for each metric of interest using the relative time column
     't_rel' and the provided steady-state window boundaries.
 
     Parameters
     ----------
     trial_df : pd.DataFrame
         Single-trial DataFrame with a 't_rel' column (seconds from trial start)
         and all standard VO2 Master output columns.
     ss_start : float
         Steady-state window start in seconds relative to trial start.
     ss_end : float
         Steady-state window end in seconds relative to trial start.
 
     Returns
     -------
     dict
         Mapping of human-readable metric name to its steady-state mean value.
         Keys: 'VO2 (mL/kg/min)', 'VO2 (mL/min)', 'VE (L/min)', 'Rf (bpm)',
               'Tv (L)', 'EE (kcal/hr)', 'FeO2 (%)', 'EqO2'.
         Values are floats; np.nan is returned for any metric with no data in window.
     """
     t = 't_rel'
     return {
         'VO2 (mL/kg/min)':  ss_mean(trial_df, t, 'VO2[mL/kg/min]',  ss_start, ss_end),
         'VO2 (mL/min)':     ss_mean(trial_df, t, 'VO2[mL/min]',      ss_start, ss_end),
         'VE (L/min)':       ss_mean(trial_df, t, 'Ve[L/min]',         ss_start, ss_end),
         'Rf (bpm)':         ss_mean(trial_df, t, 'Rf[bpm]',           ss_start, ss_end),
         'Tv (L)':           ss_mean(trial_df, t, 'Tv[L]',             ss_start, ss_end),
         'EE (kcal/hr)':     ss_mean(trial_df, t, 'Calories[kcal/hr]', ss_start, ss_end),
         'FeO2 (%)':         ss_mean(trial_df, t, 'FeO2[%]',           ss_start, ss_end),
         'EqO2':             ss_mean(trial_df, t, 'EqO2',              ss_start, ss_end),
     }
 
 
 
 
 
 # ---------------------------------------------------------------------------
 # Plotting
 # ---------------------------------------------------------------------------
def plot_timeseries_all_trials(trial_data: list) -> plt.Figure:
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
 
 
def plot_ss_bar_comparison(trial_data: list) -> plt.Figure:
     """
     Plot a grouped bar chart comparing steady-state means across trials.
 
     For each of four primary metrics (VO2, VE, Rf, EE), one bar group is
     drawn with one bar per trial. Bar heights are the steady-state mean within
     that trial's own ['win_start', 'win_end'] window (converted to
     trial-relative seconds), since each trial's window can fall at a
     different point in its timeline. Numeric labels are annotated above each
     bar. Trials with no data in the window (NaN) are silently skipped for
     that metric.
 
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
     metrics = [
         ('VO2[mL/kg/min]',    'VO₂\n(mL/kg/min)'),
         ('Ve[L/min]',          'VE\n(L/min)'),
         ('Rf[bpm]',            'Rf\n(bpm)'),
         ('Calories[kcal/hr]',  'EE\n(kcal/hr)'),
     ]
 
     n_metrics = len(metrics)
     n_trials  = len(trial_data)
     x         = np.arange(n_metrics)
     width     = 0.8 / n_trials
 
     fig, ax = plt.subplots(figsize=(10, 5))
     for i, td in enumerate(trial_data):
         df   = td['df']
         lbl  = td['label']
         clr  = TRIAL_COLORS[i % len(TRIAL_COLORS)]
         win_start_rel = td['win_start'] - td['t_start']
         win_end_rel   = td['win_end']   - td['t_start']
         vals = [ss_mean(df, 't_rel', col, win_start_rel, win_end_rel) for col, _ in metrics]
         offset = (i - n_trials / 2 + 0.5) * width
         bars = ax.bar(x + offset, vals, width, label=lbl, color=clr, edgecolor='white')
         for bar, val in zip(bars, vals):
             if not np.isnan(val):
                 ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(v for v in vals if not np.isnan(v)) * 0.01,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=9)
 
     ax.set_xticks(x)
     ax.set_xticklabels([lbl for _, lbl in metrics])
     ax.set_title('Steady-State Comparison by Trial  (each trial\'s own window)',
                  fontsize=MEDIUM_SIZE, fontweight='bold')
     ax.set_ylabel('Steady-state mean')
     ax.legend(fontsize=11)
     plt.tight_layout()
     return fig
 
 
def plot_environment(trial_data: list) -> plt.Figure:
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
 
 
# ---------------------------------------------------------------------------
# Defaults — edit these or pass CLI args
# ---------------------------------------------------------------------------
 
# ---------------------------------------------------------------------------
# Plot styling  (module level — the plot functions above reference these)
# ---------------------------------------------------------------------------
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 13, 15, 17
plt.rc('font',   size=SMALL_SIZE)
plt.rc('axes',   titlesize=SMALL_SIZE)
plt.rc('axes',   labelsize=MEDIUM_SIZE)
plt.rc('xtick',  labelsize=SMALL_SIZE)
plt.rc('ytick',  labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
COLOR_SS     = '#2CA02C'
TRIAL_COLORS = plt.cm.tab10.colors
 
# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
fPath       = "C:\\Users\\max.ferguson\\OneDrive - BOA Technology Inc\\PFL Team - General\\Testing Segments\\Outdoor\\TrailRunning\\2026_Performance_Kailas\\VO2Master\\"
OUTPUT_DIR  = fPath
fileExt     = ".xlsx"
fileExtDiag = "Diagnostics.xlsx"
fileExtHR = "HR.xlsx"
REDO_TRIALS = False   # True (or delete the *_trialseg.npy) to re-click a subject
 
MAKE_PLOTS = False    # True -> also regenerate the per-participant PNGs
 
MINUTES  = [3]     # which per-minute means become rows (the 'Time' column)
 
# ---- Column mapping — CONFIRM these against your actual VO2 file --------
COL_TIME = 'Time[s]'
COL_VO2  = 'VO2[mL/kg/min]'        # absolute VO2 (~2600 range, matches your target)
COL_EE   = 'Calories[kcal/hr]'  # see EE_SCALE note below
COL_HR   = 'HR[bpm]'            # NaN if the file has no HR column
COL_Press   = 'Pressure[hPa]'
COL_Temp   = 'Temp[C]'
EE_SCALE = 1/60.0               # kcal/hr -> kcal/min (your target EE ~13). Set 1.0 to leave as-is.
# ------------------------------------------------------------------------
 
 
# ---------------------------------------------------------------------------
# Compiled-file helpers
# ---------------------------------------------------------------------------
def subject_key(fname):
    """Common participant key, tolerant of _/-/space separators."""
    key = fname
    for suf in ('SessionDiagnostics.xlsx', 'Diagnostics.xlsx', 'HR.xlsx', '.xlsx'):
        if key.endswith(suf):
            key = key[:-len(suf)]
            break
    return key.rstrip(' _-')
 
 
def parse_config(label):
    """Return the config token (QL, SG, or BOA) from a SessionMark Details string.
 
    Matches case-insensitively so 'Boa'/'BOA'/'boa' all collapse to 'BOA'.
    Returns None if none of the known configs are present.
    """
    m = re.search(r'QL|SG|BOA', str(label), flags=re.IGNORECASE)
    return m.group(0).upper() if m else None
 
 
def summarise_window(trial_df, win_start, win_end):
    """Average over the steady-state window [win_start, win_end) in absolute time."""
    w = trial_df[(trial_df[COL_TIME] >= win_start) & (trial_df[COL_TIME] < win_end)]
    if w.empty:
        return {}
    return {
        'EE':  w[COL_EE].mean()  * EE_SCALE if COL_EE  in w else np.nan,
        'VO2': w[COL_VO2].mean()            if COL_VO2 in w else np.nan,
        'HR':  w[COL_HR].mean()             if COL_HR  in w else np.nan,
        'Pressure': w[COL_Press].mean() if COL_Press in w else np.nan,
        'Temp':     w[COL_Temp].mean()  if COL_Temp  in w else np.nan,
    }
 
 
def slice_trials(df_vo2, segments, subject, filepath):
    """Segment the recording into per-trial DataFrames.

    Trial start/end are selected once via ginput and cached to
    <subject>_trialseg.npy ({trial_key: (t0, t_end)}). Later runs load the
    cached clicks and skip manual selection. Delete the .npy or set
    REDO_TRIALS=True to re-select.
    """
    cache_path = os.path.join(filepath, subject + '_trialseg.npy')
    if os.path.exists(cache_path) and not REDO_TRIALS:
        clicks = np.load(cache_path, allow_pickle=True).item()
    else:
        clicks = {}

    trial_data = []
    updated = False
    for seg in segments:
        key = f"{seg['trial']:02d}_{seg['label']}"

        if key in clicks:                      # cached — no prompt
            t0, t_end = clicks[key]
        else:                                  # first time — select
            print(f"Select start and end for: {seg['label']} (trial {seg['trial']})")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df_vo2[COL_TIME], df_vo2[COL_VO2], label='VO2')
            ax.set_title(f"Trial {seg['trial']} — {seg['label']}\nSelect START then END")
            ax.set_xlabel('Time'); ax.set_ylabel('VO2')
            if seg['t_start'] is not None:
                ax.axvline(seg['t_start'], color='g', ls='--', alpha=0.5, label='Diag start')
            if seg['t_end'] is not None:
                ax.axvline(seg['t_end'], color='r', ls='--', alpha=0.5, label='Diag end')
            ax.legend(); plt.tight_layout()

            pts = np.asarray(plt.ginput(2, timeout=120))
            plt.close(fig)
            t0, t_end = float(pts[0, 0]), float(pts[1, 0])
            clicks[key] = (t0, t_end)
            updated = True

        mask = (df_vo2[COL_TIME] >= t0) & (df_vo2[COL_TIME] < t_end)
        tdf  = df_vo2[mask].copy()
        tdf['t_rel'] = tdf[COL_TIME] - t0

        trial_data.append({'label': seg['label'], 'trial': seg['trial'],
                           't_start': t0, 't_end': t_end, 'df': tdf,
                           'win_start': t_end - 180,
                           'win_end':   t_end - 60})

    if updated:                                # only write if new clicks happened
        np.save(cache_path, np.array(clicks, dtype=object))

    return trial_data
 
 
def process_participant(vo2_path, HR_path, diag_path, subject, check_data=False):
    """Return (tidy DataFrame, trial_data) for one participant.
    
    If check_data=True, plots VO2 for each trial and asks whether data is clean.
    Set check_data=False to skip the visual check entirely.
    """
    filepath   = os.path.dirname(vo2_path)
    
    df_vo2     = load_vo2(vo2_path)
    segments   = load_segments(diag_path)
    #%df_vo2.append(load_vo2(HR_path))
    if HR_name is None:    
        df_vo2['HR'] = 0
    else:
        filepath_hr   = os.path.dirname(HR_path)
        df_hr = load_vo2(HR_path)
        trial_data_HR = slice_trials(df_hr, segments, subject, filepath_hr)
    
    trial_data = slice_trials(df_vo2, segments, subject, filepath)
    
 
    if check_data:
        for td in trial_data:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(td['df'][COL_TIME], td['df'][COL_VO2], label='VO2')
            ax.axvline(td['t_start'], color='g', linestyle='--', label='Trial start')
            if td['t_end'] is not None:
                ax.axvline(td['t_end'], color='r', linestyle='--', label='Trial end')
 
            # Shade each steady-state minute window
            for m in MINUTES:
                ax.axvspan(td['win_start'], td['win_end'], alpha=0.2, color='orange',
                       label='Steady-state window')
 
            ax.set_title(f"{subject} | {td['label']} (trial {td['trial']})")
            ax.set_xlabel('Time')
            ax.set_ylabel('VO2')
            ax.legend()
            plt.tight_layout()
            plt.show(block=False)
            fig.canvas.manager.window.raise_()
            plt.waitforbuttonpress()  # click or keypress on the figure to advance
            plt.close(fig)
 
        answer = messagebox.askyesno("Data check", f"Is data clean for {subject}?")
        if not answer:
            raise ValueError(f"Data flagged as unclean for {subject}. Aborting.")
            
        for td in trial_data_HR:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(td['df'][COL_TIME], td['df'][COL_HR], label='HR')
            ax.axvline(td['t_start'], color='g', linestyle='--', label='Trial start')
            if td['t_end'] is not None:
                ax.axvline(td['t_end'], color='r', linestyle='--', label='Trial end')
 
            # Shade each steady-state minute window
            for m in MINUTES:
                ax.axvspan(td['win_start'], td['win_end'], alpha=0.2, color='orange',
                       label='Steady-state window')
 
            ax.set_title(f"{subject} | {td['label']} (trial {td['trial']})")
            ax.set_xlabel('Time')
            ax.set_ylabel('HR')
            ax.legend()
            plt.tight_layout()
            plt.show(block=False)
            fig.canvas.manager.window.raise_()
            plt.waitforbuttonpress()  # click or keypress on the figure to advance
            plt.close(fig)
 
        answer = messagebox.askyesno("Data check", f"Is data clean for {subject}?")
        if not answer:
            raise ValueError(f"Data flagged as unclean for {subject}. Aborting.")
 
    records = []
    for td in trial_data:
        config = parse_config(td['label'])
        order  = td['trial']
        for i in range(1, 5):
            print(i)
            row = summarise_window(td['df'], td['win_start'] + 30*i - 30,
                                              td['win_start'] + 30*i)
            if row:
                records.append({'Subject': subject, 'Config': config,
                                'Order': order, 'Q': i, **row})
    
    # --- merge HR into the matching rows ---
    if HR_name is not None:
        for td in trial_data_HR:
            config = parse_config(td['label'])
            order  = td['trial']
            for i in range(1, 5):
                row = summarise_window(td['df'], td['win_start'] + 30*i - 30,
                                                  td['win_start'] + 30*i)
                if row:
                    # find the already-built record with the same keys
                    for rec in records:
                        if (rec['Config'] == config and rec['Order'] == order
                                and rec['Q'] == i):
                            rec['HR'] = row['HR']
                            break
    
    
     
                
    return pd.DataFrame(records), trial_data
 
# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------
diag = [f for f in os.listdir(fPath) if f.endswith(fileExtDiag)]
HR = [f for f in os.listdir(fPath) if f.endswith(fileExtHR)]
data = [f for f in os.listdir(fPath)
        if f.endswith(fileExt) and not f.endswith(fileExtDiag) and not f.endswith(fileExtHR)]
 
diag_by_subject = {subject_key(d): d for d in diag}
HR_by_subject = {subject_key(d): d for d in HR}
 
all_results = []
#%%
#for i in range(1):
for vo2_name in sorted(data):
 #   vo2_name = data[4]
    subject   = subject_key(vo2_name)
    diag_name = diag_by_subject.get(subject)
    HR_name = HR_by_subject.get(subject)
    
    
    if diag_name is None:
        print(f'  ! no diagnostics file for {subject}, skipping')
        continue
    #if HR_name is None:
     #   print(f'  ! no HR file for {subject}, skipping')
      #  continue
 
    print(f'Processing {subject} ...')
    
    if HR_name is not None:
        df_sub, trial_data = process_participant(
            os.path.join(fPath, vo2_name), os.path.join(fPath, HR_name),
            os.path.join(fPath, diag_name), 
            subject)
    if HR_name is None:
        df_sub, trial_data = process_participant(
            os.path.join(fPath, vo2_name), None,
            os.path.join(fPath, diag_name), 
            subject)
    all_results.append(df_sub)
 
    if MAKE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.join(OUTPUT_DIR, "Plots", subject)  
        plot_timeseries_all_trials(trial_data).savefig(base + '_timeseries.png',  dpi=150, bbox_inches='tight')
        plot_ss_bar_comparison(trial_data).savefig(base + '_ss_compare.png',  dpi=150, bbox_inches='tight')
        plot_environment(trial_data).savefig(base + '_environment.png', dpi=150, bbox_inches='tight')
        plt.close('all')
 
compiled = pd.concat(all_results, ignore_index=True)
compiled = compiled[['Subject', 'Config', 'Order','Q', 'EE', 'VO2', 'HR', 'Pressure', 'Temp']]
 
out_csv = os.path.join(OUTPUT_DIR, 'CompiledMetabolics.csv')
compiled.to_csv(out_csv, index=False)
# compiled.to_excel(out_csv.replace('.csv', '.xlsx'), index=False)   # xlsx instead
# print(f'\nWrote {len(compiled)} rows for {compiled["Subject"].nunique()} '
     # f'participants to:\n  {out_csv}')