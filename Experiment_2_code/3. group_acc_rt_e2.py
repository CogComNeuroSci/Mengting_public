import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from itertools import combinations
import pingouin as pg

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Experiment_2_data", "beh_data"))
final_results_dir = os.path.join(current_dir, 'results')
os.makedirs(final_results_dir, exist_ok=True)

exclude_iDSi = {3, 11}
subject_dirs = sorted(
    [d for d in os.listdir(current_dir)
     if d.startswith('obs_') and int(d.split('_')[1]) not in exclude_iDSi],
    key=lambda d: int(d.split('_')[1])
)

# ============================================================
# Plot styles
# ============================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42   # Embed TrueType fonts
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.transparent'] = True
sns.set(style='white')

custom_palette = {
    'same symbols': (100/255, 142/255, 192/255),
    'same side diff symbols': (176/255, 210/255, 236/255),
    'different sides': (83/255, 58/255, 51/255)
}
colors = dict(custom_palette)
cond_order = ['same symbols', 'same side diff symbols', 'different sides']
abbr_labels = {'same symbols': 'SS', 'same side diff symbols': 'SSiDS', 'different sides': 'DSi'}
cond_palette = {k: colors[k] for k in cond_order}

TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14

# ============================================================
# Functions
# ============================================================
def style_axis(ax):
    """Show only left and bottom spines; disable grid."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(False)


def categorize_instr_type(instr_type):
    v = int(instr_type)
    if v in [0, 1, 2, 3]:
        return 'same symbols'
    elif v in [4, 5, 6, 7]:
        return 'same side diff symbols'
    elif 8 <= v <= 15:
        return 'different sides'
    else:
        return 'unknown'


def load_subject_data(subject):
    clean_dir = os.path.join(current_dir, 'clean_data')
    clean_path = os.path.join(clean_dir, f'{subject}_clean.csv')
    if not os.path.isfile(clean_path):
        print(f'[INFO] {subject} has no clean file. Skipped. ({clean_path})')
        return None

    df = pd.read_csv(clean_path, sep=';', encoding='utf-8-sig')
    df['resptime'] = pd.to_numeric(df.get('resptime', np.nan), errors='coerce')
    df['respcorrect'] = pd.to_numeric(df.get('respcorrect', np.nan), errors='coerce')
    df['condition'] = df['instr_type'].apply(categorize_instr_type)
    return df


def quick_stats(long_df, dv, cond_order, subj='sub_id', within='condition'):
    # 1) Descriptives
    desc = (long_df.groupby(within)[dv]
            .agg(mean='mean', sd='std', n='count')
            .reindex(cond_order))
    desc['sem'] = desc['sd'] / np.sqrt(desc['n'])

    # 2) rm-ANOVA
    aov = pg.rm_anova(data=long_df, dv=dv, within=within, subject=subj,
                      detailed=True, effsize='np2')
    row = aov.loc[aov['Source'] == within].iloc[0]
    aov_summary = {
        'dv': dv,
        'n_subjects': int(long_df[subj].nunique()),
        'ddof1': float(row.get('ddof1', np.nan)),
        'ddof2': float(row.get('ddof2', np.nan)),
        'F': float(row.get('F', np.nan)),
        'p': float(row.get('p-unc', np.nan)),
        'np2': float(row.get('np2', np.nan)),
    }

    # 3) Pairwise paired t-tests + Cohen's dz
    wide = long_df.pivot_table(index=subj, columns=within, values=dv, aggfunc='mean')
    pw_rows = []
    for a, b in combinations(cond_order, 2):
        pair = wide[[a, b]].dropna()
        n = pair.shape[0]

        tt = pg.ttest(pair[a], pair[b], paired=True)
        tval = float(tt['T'].iloc[0])
        pval = float(tt['p-val'].iloc[0])
        df = float(tt['dof'].iloc[0])

        diff = (pair[a] - pair[b]).to_numpy()
        sd_diff = np.std(diff, ddof=1)
        dz = float(diff.mean() / sd_diff) if sd_diff > 0 else np.nan

        pw_rows.append({
            'dv': dv, 'cond_A': a, 'cond_B': b,
            'n_pairs': n, 't': tval, 'df': df, 'p': pval,
            'cohens_dz': dz,
            'mean_A': float(pair[a].mean()),
            'mean_B': float(pair[b].mean()),
            'mean_diff_(A-B)': float(diff.mean())
        })

    pw = pd.DataFrame(pw_rows)
    return desc, aov_summary, pw


# ============================================================
# Load & compute per-subject 
# ============================================================
subject_results = []
for subject in subject_dirs:
    df = load_subject_data(subject)
    if df is None:
        print(f'[INFO] {subject} has no valid data. Skipped.')
        continue
    
    cond_acc = df.groupby('condition', dropna=False)['respcorrect'].mean().to_dict()
    cond_rt = df.groupby('condition', dropna=False)['resptime'].mean().to_dict()  

    subject_results.append({
        'subject': subject,
        'same symbols': cond_acc.get('same symbols', np.nan),
        'same side diff symbols': cond_acc.get('same side diff symbols', np.nan),
        'different sides': cond_acc.get('different sides', np.nan),
        'rt_same symbols': cond_rt.get('same symbols', np.nan),
        'rt_same side diff symbols': cond_rt.get('same side diff symbols', np.nan),
        'rt_different sides': cond_rt.get('different sides', np.nan),
    })

results_df = pd.DataFrame(subject_results)
results_df['sub_id'] = results_df['subject'].str.split('_').str[1].astype(int)
results_df.sort_values('sub_id', inplace=True, ignore_index=True)

# ============================================================
# Reshape
# ============================================================
# Accuracy (%)
acc_long = results_df.melt(
    id_vars=['subject', 'sub_id'],
    value_vars=cond_order,
    var_name='condition',
    value_name='acc'
).dropna(subset=['acc'])
acc_long['acc_pct'] = acc_long['acc'] * 100.0

# RT (ms)
rt_map = {
    'rt_same symbols': 'same symbols',
    'rt_same side diff symbols': 'same side diff symbols',
    'rt_different sides': 'different sides'
}
rt_long = results_df.melt(
    id_vars=['subject', 'sub_id'],
    value_vars=list(rt_map.keys()),
    var_name='rt_cond',
    value_name='rt_s'
).dropna(subset=['rt_s'])
rt_long['condition'] = rt_long['rt_cond'].map(rt_map)
rt_long = rt_long.drop(columns=['rt_cond'])
rt_long['rt_ms'] = rt_long['rt_s'] * 1000.0

# ============================================================
# Stats 
# ============================================================

acc_desc, acc_aov, acc_pw, acc_bal = quick_stats(acc_long, 'acc_pct', cond_order)
rt_desc, rt_aov, rt_pw, rt_bal = quick_stats(rt_long, 'rt_ms', cond_order)

# ============================================================
#  "raincloud" plot 
# ============================================================
def raincloud_half(ax, data, y_col, title, ylabel, ylims=None,
                   violin_width=0.8, mask_fudge=0.03,
                   box_shift=0.22, box_width=0.22,
                   jitter_sd=0.035, point_shift=0.22,
                   point_size=36, point_alpha=0.55,
                   box_edge_width=1.5, median_width=1.8,
                   mean_line_style='--'):
    """
    Draw: (1) left half of a violin; (2) right-shifted colored boxplot with dashed mean line;
    (3) jittered individual points above the box.
    """
    if ylims is not None:
        ax.set_ylim(*ylims)

    # 1) Full violin (to be half-masked)
    sns.violinplot(
        data=data, x='condition', y=y_col,
        order=cond_order, palette=cond_palette,
        cut=0, inner=None, width=violin_width,
        linewidth=0, ax=ax, zorder=1
    )

    # 2) Mask the right half to keep only the left side of the violin
    ymin, ymax = ax.get_ylim()
    for i, _ in enumerate(cond_order):
        ax.add_patch(Rectangle(
            (i, ymin), violin_width/2 + mask_fudge, ymax - ymin,
            facecolor='white', edgecolor='none', zorder=2, transform=ax.transData
        ))

    # 3) Colored boxplots shifted to the right
    for i, cond in enumerate(cond_order):
        ys = data.loc[data['condition'] == cond, y_col].values
        ys = ys[~np.isnan(ys)]
        if len(ys) == 0:
            continue
        bp = ax.boxplot(
            ys, positions=[i + box_shift], widths=box_width,
            vert=True, patch_artist=True, manage_ticks=False, showfliers=False, zorder=3
        )

        for box in bp['boxes']:
            box.set_facecolor('none')
            box.set_edgecolor(colors[cond])
            box.set_linewidth(box_edge_width)
            box.set_zorder(3)
        for whisker in bp['whiskers']:
            whisker.set_color(colors[cond]); whisker.set_linewidth(box_edge_width); whisker.set_zorder(3)
        for cap in bp['caps']:
            cap.set_color(colors[cond]); cap.set_linewidth(box_edge_width); cap.set_zorder(3)
        for median in bp['medians']:
            median.set_color(colors[cond]); median.set_linewidth(median_width); median.set_zorder(3)

        # Mean dashed line inside the box
        y_mean = float(np.nanmean(ys))
        x_left  = i + box_shift - box_width/2.0
        x_right = i + box_shift + box_width/2.0
        ax.hlines(y=y_mean, xmin=x_left, xmax=x_right,
                  colors=colors[cond], linestyles=mean_line_style,
                  linewidth=box_edge_width, zorder=3.2)

    # 4) Jittered subject points
    rng = np.random.default_rng(42)
    for i, cond in enumerate(cond_order):
        ys = data.loc[data['condition'] == cond, y_col].values
        ys = ys[~np.isnan(ys)]
        if len(ys) == 0:
            continue
        xs = i + point_shift + rng.normal(0, jitter_sd, size=len(ys))
        ax.scatter(xs, ys, s=point_size, alpha=point_alpha,
                   color=colors[cond], edgecolor='none', zorder=4)

    # 5) Axes and labels
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([abbr_labels[c] for c in cond_order], rotation=0, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    style_axis(ax)
    plt.tight_layout()


# ============================================================
# Figure 1: Accuracy
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 6))
raincloud_half(
    ax=ax,
    data=acc_long,
    y_col='acc_pct',
    title='Accuracy by condition',
    ylabel='Accuracy (%)',
    ylims=(45, 100),
    violin_width=0.9,
    box_shift=0.22, box_width=0.22,
    jitter_sd=0.035, point_shift=0.22,
    point_size=45, point_alpha=0.45
)
acc_pdf = os.path.join(final_results_dir, 'accuracy_raincloud_half.pdf')
plt.savefig(acc_pdf, format='pdf', bbox_inches='tight')
plt.show()

# ============================================================
# Figure 2: Response Time
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 6))
raincloud_half(
    ax=ax,
    data=rt_long,
    y_col='rt_ms',
    title='Response Time by condition',
    ylabel='Response Time (ms)',
    ylims=(350, 850),
    violin_width=0.9,
    box_shift=0.22, box_width=0.22,
    jitter_sd=0.035, point_shift=0.22,
    point_size=45, point_alpha=0.45
)
ax.set_yticks(np.arange(350, 851, 100))
rt_pdf = os.path.join(final_results_dir, 'rt_raincloud_half.pdf')
plt.savefig(rt_pdf, format='pdf', bbox_inches='tight')
plt.show()
