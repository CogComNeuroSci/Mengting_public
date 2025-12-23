import os, glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib.patches import Circle

data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Experiment_2_data"))

skip_subjects = {3, 11}
subjects = [i for i in range(1, 43) if i not in skip_subjects]
sessions = [1, 2]  # two sessions per subject
session_data = {sess: {} for sess in sessions}

eye_mov_root = os.path.join(data_root, 'beh_data', 'eye_mov_rejection')
os.makedirs(eye_mov_root, exist_ok=True)

CSV_SEP = ';'
SEP = '\t'

# Window size in px: left:0, bottom:0; right:1919, top:1079
# Square representing 2° had width 118 px → 1° = 59 px
pixel_per_degree = 59
center_coord = np.array([959.5, 539.5])  # 1919/2 and 1079/2
distance_threshold = 2  # deg

# ==========================================
# Blink detection
# ==========================================
def detect_blinks_from_xy(
    x, y,
    fs=1000,
    min_dur_ms=50,
    max_dur_ms=600,
    buffer_ms=4,
    treat_zero_as_missing=True,
    screen_bounds=None,
    lowpass_hz=30
):
    """
    Returns:
      blinks: list of (start_idx, end_idx) INCLUSIVE indices of missing stretches with blink-like duration
      x_filt, y_filt: optionally low-passed versions of x,y (same length), or the original if lowpass_hz is None
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    assert len(y) == n, "x and y must have same length"

    # ---- 1) Build a 'missing' mask ----
    missing = np.zeros(n, dtype=bool)
    if treat_zero_as_missing:
        missing |= ((x == 0) & (y == 0))
    if screen_bounds is not None:
        (xmin, xmax), (ymin, ymax) = screen_bounds
        oob = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
        missing |= oob

    # ---- 2) Find contiguous missing stretches within [min_dur, max_dur] ----
    min_dur = int(round(min_dur_ms * fs / 1000.0))
    max_dur = int(round(max_dur_ms * fs / 1000.0))
    pad     = int(round(buffer_ms * fs / 1000.0))

    blinks = []
    in_seg = False
    start = 0
    for i, m in enumerate(missing):
        if m and not in_seg:
            in_seg = True
            start = i
        elif not m and in_seg:
            end = i - 1
            dur = end - start + 1
            if min_dur <= dur <= max_dur:
                s = max(0, start - pad)
                e = min(n - 1, end + pad)
                blinks.append((s, e))
            in_seg = False
    if in_seg:
        end = n - 1
        dur = end - start + 1
        if min_dur <= dur <= max_dur:
            s = max(0, start - pad)
            e = min(n - 1, end + pad)
            blinks.append((s, e))

    # ---- 3)  NaN-safe moving average low-pass ----
    def nan_safe_moving_average(sig, win, valid_mask):
        sig = sig.astype(float)
        valid = valid_mask.copy()
        sig_ = np.zeros_like(sig)
        sig_[valid] = sig[valid]
        w = np.ones(win, dtype=float)
        num = np.convolve(sig_, w, mode='same')
        den = np.convolve(valid.astype(float), w, mode='same')
        out = num / np.maximum(den, 1e-9)
        out[den < 1] = np.nan
        return out

    if lowpass_hz is None:
        x_filt, y_filt = x.copy(), y.copy()
    else:
        win = max(3, int(round(fs / float(lowpass_hz))))
        if win % 2 == 0:
            win += 1
        valid_mask = (~missing) & np.isfinite(x) & np.isfinite(y)
        x_filt = nan_safe_moving_average(x, win, valid_mask)
        y_filt = nan_safe_moving_average(y, win, valid_mask)

    return blinks, x_filt, y_filt


def compute_dist(x, y, center_coord, pixel_per_degree):
    # Correct Euclidean distance in degrees (the original bug was a comma instead of '+')
    return np.sqrt((x - center_coord[0])**2 + (y - center_coord[1])**2) / pixel_per_degree


# ==========================================
# Main loop across subjects and sessions
# ==========================================

# Each row: [obs, session, n_trials, blink_count, eyemov_count, blink_rate, eyemov_rate, hard_count, hard_rate]
group_rows = []

for obs_i in subjects:
    for sess_i in sessions:
        et_dir = os.path.join(data_root, 'et_data', f'obs_{obs_i}', f'session_{sess_i}', 'main_task')
        beh_root = os.path.join(data_root, 'beh_data', f'obs_{obs_i}')
        beh_data_dir = os.path.join(beh_root, f'session_{sess_i}', 'main_task')

        out_dir_session = os.path.join(eye_mov_root, f'obs_{obs_i}', f'session_{sess_i}')
        os.makedirs(out_dir_session, exist_ok=True)

        events_path = os.path.join(et_dir, f'{obs_i}_events.asc')
        samples_path = os.path.join(et_dir, f'{obs_i}_samples.asc')
        if not (os.path.isfile(events_path) and os.path.isfile(samples_path)):
            print(f'[SKIP] Missing ET files for obs {obs_i} session {sess_i}:')
            print('       ', events_path)
            print('       ', samples_path)
            continue

        # ==========================================
        # EVENTS: read instruction→gratings timing per trial
        # ==========================================
        with open(events_path, 'r') as fl:
            data = fl.readlines()
        if not data:
            print(f'[SKIP] Empty events file for obs {obs_i} session {sess_i}')
            continue

        header = data.pop(0)  # keep but not used
        # parse lines
        for i in range(len(data)):
            data[i] = data[i].replace('\n', '').replace('\r', '').replace('"', '').split(SEP)

        trial_start_end_times = []
        for i in data:
            if len(i) > 1:
                if 'instructions' in i[1]:
                    trial_start_end_times.append([])
                    trial_start_end_times[-1].append(int(i[1].split(' ')[0]))
                elif 'gratings' in i[1]:
                    trial_start_end_times[-1].append(int(i[1].split(' ')[0]))

        # ==========================================
        # SAMPLES: read x,y,pupil over time; make per-trial series
        # ==========================================
        with open(samples_path, 'r') as fl:
            raw_data = fl.readlines()

        raw_datarr = np.zeros([len(raw_data), 4])
        for ind, line in enumerate(raw_data):
            parts = line.split('\t')
            if len(parts) > 1 and parts[1].strip() != '.':
                raw_datarr[ind, :] = np.array(parts[:-1]).astype(float)
            else:
                raw_datarr[ind, :] = [int(parts[0]), 0, 0, 0]

        trial_gaze_dist, trial_gaze_xy = [], []
        for ind, (t0, t1) in enumerate(trial_start_end_times):
            if not (ind % 20):
                print(f'obs {obs_i} sess {sess_i} - parsing trial {ind}')
            mask = (raw_datarr[:, 0] >= t0) & (raw_datarr[:, 0] <= t1)
            trial_gaze_dist.append(
                compute_dist(
                    x=raw_datarr[mask, 1],
                    y=raw_datarr[mask, 2],
                    center_coord=center_coord,
                    pixel_per_degree=pixel_per_degree
                )
            )
            trial_gaze_xy.append(raw_datarr[mask, 1:3])

        n_trials = len(trial_gaze_dist)
        if n_trials == 0:
            print(f'[SKIP] No trials parsed for obs {obs_i} session {sess_i}')
            continue

        # ==========================================
        # BLINKS (trial-level)
        # ==========================================
        blinks_all = []
        for i in range(n_trials):
            blinks, x_lp, y_lp = detect_blinks_from_xy(
                trial_gaze_xy[i][:, 0], trial_gaze_xy[i][:, 1],
                fs=1000, min_dur_ms=50, max_dur_ms=600, buffer_ms=4,
                treat_zero_as_missing=True, screen_bounds=None, lowpass_hz=30
            )
            blinks_all.append(blinks)

        blink_trials = np.array([len(b) > 0 for b in blinks_all], dtype=bool)

        # ==========================================
        # EYEMOV: check outside ALL blink segments
        # ==========================================
        eyemov = []
        for t in range(n_trials):
            n = len(trial_gaze_dist[t])
            no_blink_mask = np.ones(n, dtype=bool)
            for (s, e) in blinks_all[t]:
                no_blink_mask[s:e+1] = False  # inclusive interval
            eyemov.append(np.any(trial_gaze_dist[t][no_blink_mask] > distance_threshold))
        eyemov = np.array(eyemov, dtype=bool)

        # HardFilter: out-of-fixation OR blink
        reject_legacy = (eyemov | blink_trials)

        # ==========================================
        # Plot & save per-session diagnostic figure under obs_x/session_y
        # ==========================================
        fig, axs = pl.subplots(1, 2, figsize=(12, 5))
        for trial_i in range(n_trials):
            col = ['g', 'r'][int(eyemov[trial_i])]
            axs[0].plot(trial_gaze_dist[trial_i], '%s-' % col,
                        zorder=[2, 1][int(eyemov[trial_i])])
            axs[1].plot(trial_gaze_xy[trial_i][:, 0], trial_gaze_xy[trial_i][:, 1],
                        '%s-' % col, alpha=.5, zorder=[2, 1][int(eyemov[trial_i])])
        axs[1].set_xlim([0, 1919])
        axs[1].set_ylim([0, 1079])
        c = Circle(center_coord, radius=distance_threshold * pixel_per_degree, ec='b', fill=False, zorder=0)
        axs[1].add_artist(c)
        pl.suptitle(
            'obs %i - session %i\ngreen: ok trials, red: rejected trials (%.2f percent)'
            % (obs_i, sess_i, 100 * np.mean(eyemov))
        )
        axs[0].set_xlabel('Time from instruction onset (EyeT samples)')
        axs[0].set_ylabel('Distance from fixation (deg)')
        axs[1].set_xlabel('Screen X (pixels)')
        axs[1].set_ylabel('Screen Y (pixels)')

        fig_path = os.path.join(out_dir_session, f'eyemov_overview_obs{obs_i}_sess{sess_i}.png')
        pl.savefig(fig_path, dpi=150, bbox_inches='tight')
        pl.close(fig)

        # ==========================================
        # Compare with behavior & SAVE per-session CSV 
        # ==========================================
        beh_files = glob.glob(os.path.join(beh_data_dir, f'task_obs{obs_i}_*.txt'))
        df_beh = None
        if len(beh_files) > 0:
            df_beh = pd.read_csv(beh_files[0], delimiter='\t')

        print(
            'obs %i - session %i\n\t%i trials with eyemov (%.2f%%) - blink trials %i (%.2f%%)\n\tNOT BLINK but rejected by eyemov: %i\n'
            % (
                obs_i, sess_i,
                int(eyemov.sum()), 100 * np.mean(eyemov),
                int(blink_trials.sum()), 100 * np.mean(blink_trials),
                int((eyemov & ~blink_trials).sum())
            )
        )

        out_csv = os.path.join(out_dir_session, 'trials_to_reject_eyeT.csv')
        if df_beh is not None and len(df_beh) == len(eyemov):
            print('\tNICE! Number of trials match (%i). DATA SAVED!' % len(df_beh))
            df_results = pd.DataFrame(
                np.vstack([
                    np.repeat(obs_i, len(df_beh)),
                    np.repeat(sess_i, len(df_beh)),
                    np.arange(len(df_beh)),
                    reject_legacy.astype(int),   # HardFilter (legacy OR)
                    eyemov.astype(int),          # SoftFilter (eyemov only)
                    blink_trials.astype(int)
                ]).T,
                columns=['obs', 'session', 'trial_num', 'reject_legacy', 'eyemov', 'blink_trial']
            )
            df_results.to_csv(out_csv, index=False, sep=CSV_SEP, encoding='utf-8-sig')
        else:
            # If behavior is missing or count mismatch, still save with available length (trial_num = ET index)
            if df_beh is None:
                print('\t[WARN] Behavior file missing. Saving ET-only results with ET trial count.')
            else:
                print('\t[WARN] EyeT (%i) vs Behav (%i) trial counts DO NOT match. Saving ET-only results.' % (len(eyemov), len(df_beh)))
            df_results = pd.DataFrame(
                np.vstack([
                    np.repeat(obs_i, n_trials),
                    np.repeat(sess_i, n_trials),
                    np.arange(n_trials),
                    reject_legacy.astype(int),
                    eyemov.astype(int),
                    blink_trials.astype(int)
                ]).T,
                columns=['obs', 'session', 'trial_num', 'reject_legacy', 'eyemov', 'blink_trial']
            )
            df_results.to_csv(out_csv, index=False, sep=CSV_SEP, encoding='utf-8-sig')

        # ==========================================
        # group-level summary 
        # ==========================================
        blink_count = int(blink_trials.sum())      # # trials with at least one blink
        eyemov_count = int(eyemov.sum())           # # trials with out-of-fixation >2° (outside all blink segments)
        hard_count = int(reject_legacy.sum())      # # trials rejected by HardFilter (blink OR eyemov)

        blink_rate = float(blink_count) / n_trials
        eyemov_rate = float(eyemov_count) / n_trials
        hard_rate  = float(hard_count)  / n_trials

        group_rows.append([obs_i, sess_i, n_trials, blink_count, eyemov_count, blink_rate, eyemov_rate, hard_count, hard_rate])

# ==========================================
# Save summary 
# ==========================================

df_group = pd.DataFrame(
    group_rows,
    columns=['obs', 'session', 'n_trials', 'blink_count', 'eyemov_count', 'blink_rate', 'eyemov_rate', 'hard_count', 'hard_rate']
)
group_csv = os.path.join(eye_mov_root, 'group_level_blink_eyemov_rates.csv')
df_group.to_csv(group_csv, index=False, sep=CSV_SEP, encoding='utf-8-sig')
print(f'[DONE] Group-level summary saved to: {group_csv}')


