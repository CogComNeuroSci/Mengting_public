import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binomtest

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Experiment_2_data", "beh_data"))
eyedir      = os.path.join(current_dir, "eye_mov_rejection")
clean_dir   = os.path.join(current_dir, "clean_data")
if not os.path.exists(clean_dir):
    os.makedirs(clean_dir, exist_ok=True)

ET_FILENAME          = "trials_to_reject_eyeT.csv"
ALPHA                = 0.05                           # Binomial threshold
HARD_RATE_THRESH     = 0.5                            # Subject-level hard-rate cutoff

# ============================================================
# Only process obs_1 .. obs_42, and skip obs_3 and obs_11
# ============================================================

SKIP_SUBJECTS = {3, 11}

subject_dirs = [
    e for e in os.scandir(current_dir)
    if e.is_dir()
    and e.name.startswith("obs_")
    and e.name[4:].isdigit()
    and 1 <= int(e.name[4:]) <= 42
    and int(e.name[4:]) not in SKIP_SUBJECTS
]

subject_dirs.sort(key=lambda e: int(e.name[4:]))

# ----------------------------------------------
# Stage A: Compute subject-level mean hard-rate from per-session ET files
# ----------------------------------------------
subj_to_mean = {}
for d in subject_dirs:
    try:
        obs_id = int(d.name.split('_')[1])
    except Exception:
        continue

    session_rates = []
    for sess in [1, 2]:
        et_path = os.path.join(eyedir, f"obs_{obs_id}", f"session_{sess}", ET_FILENAME)
        et_df = pd.read_csv(et_path, sep=';', encoding='utf-8')
        et_df['reject_legacy'] = pd.to_numeric(et_df['reject_legacy'], errors='coerce')
        session_rates.append(float(et_df['reject_legacy'].mean()))

    if len(session_rates) < 2:
        print(f"[WARN] obs={obs_id}: only {len(session_rates)} ET session(s) found")

    subj_to_mean[obs_id] = float(np.nanmean(session_rates)) if session_rates else np.nan

drop_subjects = [int(k) for k, v in subj_to_mean.items() if (not np.isnan(v)) and (v > HARD_RATE_THRESH)]


# ----------------------------------------------
# Stage B: binomial per session -> clean missing trials
# ----------------------------------------------
summary_rows = []

for d in subject_dirs:
    name_parts = d.name.split('_')
    try:
        obs_id = int(name_parts[1])
    except Exception:
        print(f"[SKIP] irregular subject directory: {d.name}")
        continue

    if obs_id in drop_subjects:
        print(f"[DROP-WHOLE] obs={obs_id} by subject-level hard-rate > {HARD_RATE_THRESH}")
        summary_rows.append({
            'obs': obs_id,
            'dropped_by_hard_rate': True,
            'subject_hard_rate_avg_orig': subj_to_mean.get(obs_id, np.nan),
            'n_sessions_kept': 0,
            'n_trials_kept': 0,
            'binomial_kept_sessions': "[]",
        })
        continue

    kept_frames = []
    binomial_kept_sessions = []

    for sess in [1, 2]:
        beh_path = None
        sess_dir = os.path.join(d.path, f"session_{sess}", "main_task")
        if os.path.isdir(sess_dir):
            files = glob.glob(os.path.join(sess_dir, '*.txt'))
            if files:
                beh_path = files[0]

        if beh_path is None:
            rec_root = os.path.join(d.path, f"session_{sess}")
            files = glob.glob(os.path.join(rec_root, '**', '*.txt'), recursive=True)
            if files:
                beh_path = files[0]

        if beh_path is None:
            print(f"[SKIP] obs={obs_id} session={sess}: no behavior file found.")
            continue


        df_beh = pd.read_csv(beh_path, sep='\t', encoding='utf-8', engine='python')

        df_beh['respcorrect'] = pd.to_numeric(df_beh['respcorrect'], errors='coerce')
        df_beh['resptime']    = pd.to_numeric(df_beh['resptime'], errors='coerce')

        m_resp = df_beh['response'].astype(str).str.lower() != 'none'
        m_rt   = df_beh['resptime'] != -1
        m_acc  = df_beh['respcorrect'].notna()
        valid_mask = (m_resp & m_rt & m_acc)

        n_total = int(valid_mask.sum())
        n_corr  = int(df_beh.loc[valid_mask, 'respcorrect'].sum()) if n_total > 0 else 0

        pval = binomtest(n_corr, n_total, p=0.5, alternative='greater').pvalue
        if pval >= ALPHA:
            print(f"[REJECT-SESSION][Binomial] obs={obs_id} session={sess}: n_total={n_total} n_correct={n_corr} p={pval:.4g}")
            continue

        df_keep = df_beh.loc[valid_mask].copy()
        df_keep['observer'] = obs_id
        df_keep['session']  = sess

        kept_frames.append(df_keep)
        binomial_kept_sessions.append(sess)

    # Save per-subject cleaned CSV
    if kept_frames:
        out_df = pd.concat(kept_frames, ignore_index=True)
        out_path = os.path.join(clean_dir, f"obs_{obs_id}_clean.csv")
        out_df.to_csv(out_path, sep=';', index=False, encoding='utf-8-sig')

        summary_rows.append({
            'obs': obs_id,
            'dropped_by_hard_rate': False,
            'subject_hard_rate_avg_orig': subj_to_mean.get(obs_id, np.nan),
            'n_sessions_kept': len(binomial_kept_sessions),
            'n_trials_kept': len(out_df),
            'binomial_kept_sessions': str(binomial_kept_sessions),
        })
    else:
        summary_rows.append({
            'obs': obs_id,
            'dropped_by_hard_rate': False,
            'subject_hard_rate_avg_orig': subj_to_mean.get(obs_id, np.nan),
            'n_sessions_kept': 0,
            'n_trials_kept': 0,
            'binomial_kept_sessions': "[]",
        })


# ----------------------------------------------
# Save cleaning summary
# ----------------------------------------------
summary = pd.DataFrame(summary_rows, columns=[
    'obs', 'dropped_by_hard_rate', 'subject_hard_rate_avg_orig',
    'n_sessions_kept', 'n_trials_kept', 'binomial_kept_sessions'
])

sum_path = os.path.join(clean_dir, 'cleaning_summary.csv')
summary.to_csv(sum_path, sep=';', index=False, encoding='utf-8-sig')

