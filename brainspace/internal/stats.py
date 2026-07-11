"""Statistical analysis over Experiment grid-search results.

Two responsibilities:
  1. Partition a trained Processor's test loss into in-distribution vs.
     out-of-distribution subsets of the existing (random-portion) test set,
     alongside the full aggregate (``compute_subset_test_losses``).
  2. Given all trials' results, run an LMM across all IVs, a Friedman +
     Nemenyi ranking per non-ordinal IV, and a Spearman correlation per
     ordinal IV -- each per data view -- and render the result as a
     markdown section (``run_statistical_tests``).

Subset definitions (see also README.md "Grid Search" section):
  - all:        the full random-portion test set (matches final_test_loss).
  - in_domain:  rows in-distribution on BOTH axes -- every valid input value
                inside the training ``x_range`` AND sequence length within the
                training range (i.e. len_id AND domain_id).
  - domain_id:  rows with every valid input value inside the training
                ``x_range``, regardless of sequence length (the complement of
                domain_ood). Populated even when the length axis is pushed OOD.
  - len_id:     rows whose valid sequence length is within the training range,
                regardless of input magnitude (the complement of len_ood). NaN
                for non-padded (fixed-length) datasets, where length doesn't
                vary. Populated even when the domain axis is pushed OOD.
  - domain_ood: rows with at least one valid input value outside
                ``x_range`` (the training domain). Not the complement of
                in_domain alone -- overlaps with len_ood.
  - len_ood:    rows whose valid sequence length exceeds the longest
                sequence length seen in training. NaN for non-padded
                (fixed-length) datasets, where this concept doesn't apply.

``in_domain`` = ``len_id`` ∩ ``domain_id``; splitting the two axes out lets a
dataset that pushes both magnitude and length OOD at once (so the strict
``in_domain`` intersection is near-empty) still be analyzed along each axis
separately.
"""
import math
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

try:
    import scikit_posthocs as sp
except ImportError:  # pragma: no cover - exercised only if dependency missing
    sp = None

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover
    smf = None


_EPS = 1e-12
_DATA_VIEWS = [
    ('all', 'All Data'),
    ('in_domain', 'In-Domain (len & domain)'),
    ('domain_id', 'Domain-In-Dist'),
    ('len_id', 'Length-In-Dist'),
    ('domain_ood', 'Domain-OOD'),
    ('len_ood', 'Length-OOD'),
]

# dataviz skill palette (references/palette.md): single-hue sequential blue
# ramp for magnitude, 'good' status green for the best/significant highlight.
_SEQ_RAMP = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#0d366b'])
_GOOD_COLOR = '#0ca30c'
_MUTED_COLOR = '#898781'
# Effect-size gradient cap for the p-value/effect-size heatmap: |paired
# Cohen's d| >= this shows fully saturated (a "very large" effect by
# conventional benchmarks, well past the "large" d=0.8 threshold).
_EFFECT_SIZE_CAP = 3.0
# Fixed categorical hue order (dataviz skill palette.md) -- never cycled/reassigned.
_CATEGORICAL = ['#2a78d6', '#1baf7a', '#eda100', '#008300',
                 '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']


def compute_subset_test_losses(processor) -> dict:
    """Partition final-epoch test loss into in-/out-of-distribution subsets of
    the existing random-portion test set, plus the 'all' aggregate.

    Returns a dict with keys 'all', 'in_domain', 'domain_id', 'len_id',
    'domain_ood', 'len_ood' (float MSE, or ``nan`` when a subset selects zero
    rows / doesn't apply). ``domain_id``/``len_id`` are the per-axis
    in-distribution complements of ``domain_ood``/``len_ood``; ``in_domain`` is
    their intersection (see module docstring).
    """
    model = processor.model
    model.eval()
    n = processor._x_test_random_n
    lo, hi = processor.x_range

    with torch.no_grad():
        x_clean = torch.nan_to_num(processor.x_test, nan=0.0)
        out, _ = processor._forward_in_batches(x_clean, processor.test_mask)

        x_n = processor.x_test[:n]
        mask_n = processor.test_mask[:n].bool()
        seq_lens = mask_n.long().sum(dim=1)
        train_max_len = max(processor.train_sequence_lengths)

        # Any valid (non-padded) timestep/feature outside the training domain.
        oob = ((x_n < lo) | (x_n > hi)) & mask_n.unsqueeze(-1)
        domain_ood_mask = oob.any(dim=(1, 2))
        len_ood_mask = (
            seq_lens > train_max_len if processor.use_padding
            else torch.zeros_like(seq_lens, dtype=torch.bool)
        )
        in_domain_mask = ~(domain_ood_mask | len_ood_mask)
        # Per-axis in-distribution complements: domain_id ignores length,
        # len_id ignores magnitude, so each stays populated even when the
        # *other* axis is pushed OOD for the whole test set.
        domain_id_mask = ~domain_ood_mask
        len_id_mask = (
            ~len_ood_mask if processor.use_padding
            else torch.zeros_like(seq_lens, dtype=torch.bool)
        )

        masks = {
            'in_domain': in_domain_mask,
            'domain_id': domain_id_mask,
            'len_id': len_id_mask,
            'domain_ood': domain_ood_mask,
            'len_ood': len_ood_mask,
        }

        if getattr(model, 'seq2seq', False):
            out_seq = out.squeeze(-1)[:n]
            valid = processor.test_cot_mask[:n]
            targets = torch.where(
                valid.bool(), processor.y_test_intermediate[:n],
                torch.zeros_like(processor.y_test_intermediate[:n]),
            )
            step_losses = F.mse_loss(out_seq, targets, reduction='none')  # (n, T)

            def subset_loss(row_mask):
                sv = valid * row_mask.unsqueeze(-1).float()
                denom = sv.sum()
                if denom.item() == 0:
                    return float('nan')
                return ((step_losses * sv).sum() / denom).item()
        else:
            out_n = out.squeeze()[:n]
            y_n = processor.y_test[:n]
            per_row = F.mse_loss(out_n, y_n, reduction='none')
            if per_row.dim() > 1:
                per_row = per_row.mean(dim=tuple(range(1, per_row.dim())))

            def subset_loss(row_mask):
                if row_mask.sum().item() == 0:
                    return float('nan')
                return per_row[row_mask].mean().item()

        result = {'all': float(processor.logs['test_loss'][-1])}
        for name, m in masks.items():
            result[name] = subset_loss(m)
        if not processor.use_padding:
            # Length axis doesn't vary → both length views are undefined.
            result['len_ood'] = float('nan')
            result['len_id'] = float('nan')
    return result


def _iv_column(iv_key: str, sample_row: dict):
    """Map an Experiment IV key (possibly dot-notation) to its result_entry
    column name (``config_{sub}_{field}``), matching Experiment._apply_iv's
    sub-config search order (data, model, train)."""
    if '.' in iv_key:
        sub_name, attr = iv_key.split('.')
        col = f'config_{sub_name}_{attr}'
        return col if col in sample_row else None
    for sub_name in ('data', 'model', 'train'):
        col = f'config_{sub_name}_{iv_key}'
        if col in sample_row:
            return col
    return None


def _successful_rows(results) -> list:
    """Rows with a valid (non-nan) test_loss_all, the common precondition for
    any of the IV-level statistics below."""
    return [r for r in results if r.get('test_loss_all') is not None
            and not (isinstance(r.get('test_loss_all'), float) and np.isnan(r['test_loss_all']))]


def _prepare_iv_frame(successes, ivs, ordinal_ivs):
    """Build the DataFrame + IV-column bookkeeping shared by
    ``run_statistical_tests`` and the per-IV/per-pair PNG renderers: a
    results.csv-row DataFrame with a ``log_error_<view>`` column per data
    view, plus which IV keys map to which (varying) result column and how
    they split into ordinal/non-ordinal. Assumes ``successes`` already passed
    the caller's own count/IV-presence checks."""
    df = pd.DataFrame(successes)
    sample_row = successes[0]

    # Dedupe by resolved column, not by key label: a bare key ('hidden_dim')
    # and its dotted form ('model.hidden_dim') both resolve to the same
    # config_* column when one is user-declared and the other is inferred
    # from column names (write_summary_md's _infer_varied_ivs fallback) --
    # without this, the same column gets analyzed twice under two labels.
    iv_cols = {}
    seen_cols = set()
    for key in ivs:
        col = _iv_column(key, sample_row)
        if col is not None and col not in seen_cols and df[col].nunique() > 1:
            iv_cols[key] = col
            seen_cols.add(col)
    non_ordinal_ivs = [k for k in iv_cols if k not in ordinal_ivs]
    ordinal_iv_cols = [(k, iv_cols[k]) for k in ordinal_ivs if k in iv_cols]

    for view_key, _ in _DATA_VIEWS:
        loss_col = f'test_loss_{view_key}'
        err_col = f'log_error_{view_key}'
        # A view's loss column is absent entirely when every row predates it
        # (e.g. analyzing a store written before len_id/domain_id existed);
        # treat it as all-NaN so downstream renders "no data" rather than
        # raising. Mixed old+new stores get the column with blanks → NaN.
        if loss_col in df.columns:
            df[err_col] = np.log(np.clip(df[loss_col].astype(float), _EPS, None))
        else:
            df[err_col] = np.nan

    return df, iv_cols, non_ordinal_ivs, ordinal_iv_cols


def run_statistical_tests(results, ivs, ordinal_ivs=None) -> str:
    """Run LMM / Friedman+Nemenyi / Spearman over experiment results and
    render the findings as a markdown section (returned as a string)."""
    ordinal_ivs = list(ordinal_ivs or [])

    successes = _successful_rows(results)
    if not ivs:
        return "## Statistical Analysis\n\n_No IVs were varied in this grid search; skipping statistical tests._\n\n"
    if len(successes) < 4:
        return ("## Statistical Analysis\n\n_Too few successful trials "
                f"({len(successes)}) to run statistical tests._\n\n")

    df, iv_cols, non_ordinal_ivs, ordinal_iv_cols = _prepare_iv_frame(successes, ivs, ordinal_ivs)

    out = ["## Statistical Analysis\n\n"]
    out.append(_render_headline(df, iv_cols, non_ordinal_ivs, ordinal_iv_cols))
    out.append(_render_lmm(df, iv_cols, non_ordinal_ivs, ordinal_iv_cols))
    for view_key, view_label in _DATA_VIEWS:
        err_col = f'log_error_{view_key}'
        if df[err_col].isna().all():
            out.append(f"### {view_label}\n\n_No data in this subset (e.g. no padded/variable-length "
                       "configs in this grid); skipping.\n\n")
            continue
        out.append(f"### {view_label}\n\n")
        out.append(_render_friedman_nemenyi(df, err_col, non_ordinal_ivs, iv_cols))
        out.append(_render_spearman(df, err_col, ordinal_iv_cols, iv_cols))
    return "".join(out)


def _slug(key: str) -> str:
    """Filesystem-safe stand-in for an IV key (e.g. 'model.hidden_dim' ->
    'model_hidden_dim'), for per-IV/per-pair output filenames."""
    return re.sub(r'[^0-9A-Za-z_]+', '_', key)


def render_iv_pairs_violin_pngs(results, ivs, ordinal_ivs, output_dir) -> list:
    """Render one PNG per unordered pair of varying IVs (x-axis = first IV's
    levels, hue = second IV's levels, y = All-Data log error), with one panel
    per stratum of the *other* varying IVs -- e.g. 'arch vs cot' gets one
    panel per hidden_dim value, not a single value pooled across hidden_dim.
    Mirrors _render_friedman_nemenyi's per-stratum markdown breakdown so the
    images and text never disagree about what was pooled vs. split out.
    Grouped/dodged via matplotlib's native violinplot (no seaborn dependency).

    Returns a list of (pair_label, relpath) for each pair that had at least
    one non-empty stratum; empty list if nothing could be rendered (including
    <2 varying IVs). No files are written for skipped pairs.
    """
    import itertools
    from .. import visualization  # local import: avoids a hard dependency for callers that never render

    ordinal_ivs = list(ordinal_ivs or [])
    successes = _successful_rows(results)
    if not ivs or len(successes) < 4:
        return []

    df, iv_cols, non_ordinal_ivs, ordinal_iv_cols = _prepare_iv_frame(successes, ivs, ordinal_ivs)
    if len(iv_cols) < 2:
        return []

    err_col = 'log_error_all'
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    for (x_key, x_col), (hue_key, hue_col) in itertools.combinations(iv_cols.items(), 2):
        other_iv_cols = [c for k, c in iv_cols.items() if k not in (x_key, hue_key)]
        strata = list(_strata(df, other_iv_cols))
        n = len(strata)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        with visualization._RENDER_LOCK:
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for ax, (stratum_label, sub_df) in zip(axes_flat, strata):
                _plot_violin_pair(ax, sub_df, x_col, hue_col, err_col, x_key, hue_key,
                                  stratum_label or "(all)")

            for ax in axes_flat[n:]:
                ax.set_visible(False)

            fig.suptitle(f"`{x_key}` vs `{hue_key}` (All-Data log error, per stratum)", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fname = f"iv_pairs_violin_{_slug(x_key)}_vs_{_slug(hue_key)}.png"
            output_path = os.path.join(output_dir, fname)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        outputs.append((f"{x_key} vs {hue_key}", os.path.basename(output_path)))
    return outputs


def _plot_violin_pair(ax, df, x_col, hue_col, err_col, x_key, hue_key, title):
    """One grouped/dodged violin panel (a single stratum): x-axis = x_col's
    levels, one violin per hue_col level (fixed categorical colors) offset
    within each x group. Cells with <2 points fall back to a scatter strip so
    violinplot never raises on a too-small sample."""
    x_levels = sorted(df[x_col].dropna().unique(), key=str)
    hue_levels = sorted(df[hue_col].dropna().unique(), key=str)
    width = 0.8 / max(len(hue_levels), 1)

    for h_idx, h_val in enumerate(hue_levels):
        color = _CATEGORICAL[h_idx % len(_CATEGORICAL)]
        offset = (h_idx - (len(hue_levels) - 1) / 2) * width
        positions, data = [], []
        for x_idx, x_val in enumerate(x_levels):
            cell = df[(df[x_col] == x_val) & (df[hue_col] == h_val)][err_col].dropna()
            if len(cell) == 0:
                continue
            positions.append(x_idx + offset)
            data.append(cell.values)

        for pos, vals in zip(positions, data):
            if len(vals) < 2:
                ax.scatter([pos] * len(vals), vals, color=color, s=14, zorder=3)
                continue
            parts = ax.violinplot([vals], positions=[pos], widths=width * 0.9,
                                    showmeans=True, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_alpha(0.7)
                body.set_edgecolor(color)
            parts['cmeans'].set_color(color)

    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([str(v) for v in x_levels])
    ax.set_xlabel(f"`{x_key}`")
    ax.set_ylabel("log error")
    ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([0], [0], color=_CATEGORICAL[i % len(_CATEGORICAL)], lw=6)
               for i in range(len(hue_levels))]
    ax.legend(handles, [str(v) for v in hue_levels], title=hue_key, fontsize=8, title_fontsize=8)


def render_pvalue_heatmap_pngs(results, ivs, ordinal_ivs, output_dir) -> list:
    """Render one PNG per non-ordinal varying IV, with one pairwise
    p-value/effect-size heatmap panel per stratum of the *other* varying IVs
    (mirrors render_iv_pairs_violin_pngs' per-stratum split and
    _render_friedman_nemenyi's markdown breakdown, rather than pooling).

    Each cell shows both numbers (p-value + stars, paired Cohen's d), but the
    *gradient* encodes effect-size magnitude, not significance: only cells
    passing p<0.05 are shaded by |d| (darker = larger effect); non-significant
    cells render as a flat muted gray regardless of |d|, so a huge but
    noisy/underpowered effect never reads as visually "strong". Each panel
    title includes the adjacent-comparison ranking chain.

    Returns a list of (iv_label, relpath) for IVs with at least one
    renderable stratum; empty list if nothing could be rendered. No files are
    written for skipped IVs.
    """
    from .. import visualization  # local import: avoids a hard dependency for callers that never render

    ordinal_ivs = list(ordinal_ivs or [])
    successes = _successful_rows(results)
    if not ivs or len(successes) < 4:
        return []

    df, iv_cols, non_ordinal_ivs, ordinal_iv_cols = _prepare_iv_frame(successes, ivs, ordinal_ivs)
    if not non_ordinal_ivs:
        return []

    err_col = 'log_error_all'
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    neutral_rgba = to_rgba('#e1e0d9')       # diagonal (self-comparison)
    muted_rgba = to_rgba(_MUTED_COLOR, 0.35)  # not significant

    for key in non_ordinal_ivs:
        col = iv_cols[key]
        other_iv_cols = [c for k, c in iv_cols.items() if k != key]
        panels = []
        for stratum_label, sub_df in _strata(df, other_iv_cols):
            pivot = _friedman_pivot(sub_df, col, ['trial_idx'], err_col)
            if pivot.shape[0] < 3 or pivot.shape[1] < 2:
                continue
            ranks, pairwise, _, _, _ = _rank_test(pivot)
            if pairwise is None:
                continue
            effect = _effect_size_matrix(pivot)
            panels.append((stratum_label or "(all)", ranks, pairwise, effect))

        if not panels:
            continue

        n = len(panels)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        with visualization._RENDER_LOCK:
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for ax, (stratum_label, ranks, pairwise, effect) in zip(axes_flat, panels):
                levels = list(pairwise.columns)
                m = len(levels)
                p_mat = pairwise.values.astype(float)
                d_mat = effect.loc[levels, levels].values.astype(float)

                colors = np.empty((m, m, 4))
                for i in range(m):
                    for j in range(m):
                        if i == j:
                            colors[i, j] = neutral_rgba
                        elif p_mat[i, j] < 0.05:
                            t = min(abs(d_mat[i, j]), _EFFECT_SIZE_CAP) / _EFFECT_SIZE_CAP
                            colors[i, j] = _SEQ_RAMP(t)
                        else:
                            colors[i, j] = muted_rgba
                ax.imshow(colors)
                ax.set_xticks(range(m))
                ax.set_yticks(range(m))
                ax.set_xticklabels([str(l) for l in levels], rotation=45, ha='right')
                ax.set_yticklabels([str(l) for l in levels])
                for i in range(m):
                    for j in range(m):
                        if i == j:
                            continue
                        p, d = p_mat[i, j], d_mat[i, j]
                        d_str = f"{d:+.2f}" if np.isfinite(d) else ("+inf" if d > 0 else "-inf")
                        sig = p < 0.05
                        t = min(abs(d), _EFFECT_SIZE_CAP) / _EFFECT_SIZE_CAP if sig else 0.0
                        text_color = 'white' if (sig and t > 0.55) else 'black'
                        ax.text(j, i, f"d={d_str}\np={p:.2g}{_pvalue_stars(p)}", ha='center',
                                va='center', fontsize=7, color=text_color)
                chain = _ranking_chain(ranks, pairwise)
                ax.set_title(f"{stratum_label}\n{chain}", fontsize=10)
                sm = ScalarMappable(cmap=_SEQ_RAMP, norm=Normalize(0, _EFFECT_SIZE_CAP))
                fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="|paired Cohen's d| (p<0.05 only)")

            for ax in axes_flat[n:]:
                ax.set_visible(False)

            fig.suptitle(f"`{key}` Pairwise Effect Size (gray = not significant, p≥0.05)", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fname = f"iv_pvalue_heatmap_{_slug(key)}.png"
            output_path = os.path.join(output_dir, fname)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        outputs.append((key, os.path.basename(output_path)))
    return outputs


def _friedman_pivot(df, treatment_col, block_cols, value_col):
    """Pivot df to blocks x treatment-levels for a Friedman design, averaging
    over duplicate (block, treatment) rows and dropping incomplete blocks."""
    grouped = df.groupby(block_cols + [treatment_col])[value_col].mean().reset_index()
    pivot = grouped.pivot_table(index=block_cols, columns=treatment_col, values=value_col)
    return pivot.dropna(axis=0, how='any')


def _effect_size_matrix(pivot):
    """Paired Cohen's d_z (mean(diff) / std(diff, ddof=1)) for every ordered
    pair of pivot columns, diff = level_j - level_i matched by block (trial)
    -- positive means level_j has higher (worse) log error than level_i.
    Antisymmetric: entry (i, j) == -entry (j, i). A ~zero paired std (every
    block agrees on direction) maps to +/-inf rather than raising."""
    cols = list(pivot.columns)
    n = len(cols)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = (pivot[cols[j]] - pivot[cols[i]]).values
            sd = diff.std(ddof=1)
            if sd < _EPS:
                m = diff.mean()
                mat[i, j] = np.inf if m > 0 else (-np.inf if m < 0 else 0.0)
            else:
                mat[i, j] = diff.mean() / sd
    return pd.DataFrame(mat, index=cols, columns=cols)


def _mean_ranks(pivot):
    """Mean rank per treatment (ascending: rank 1 = lowest/best log error)."""
    return pivot.rank(axis=1, method='average').mean(axis=0).sort_values()


def _rank_test(pivot):
    """Rank treatment levels in a blocks x treatment-levels pivot (>=2 columns)
    and test whether they differ, choosing the test that applies:

    - Exactly 2 levels: a paired Wilcoxon signed-rank test (Friedman's
      chi-square approximation is undefined below 3 treatments -- scipy
      raises on it -- so 2-level IVs get the natural paired-test fallback
      instead of an error).
    - 3+ levels: Friedman omnibus test, then scikit-posthocs' Nemenyi
      post-hoc for pairwise significance (if scikit-posthocs is installed).

    Returns (ranks, pairwise_pvals_df_or_None, test_name, stat, pval). Any
    failure surfaces as pval=nan / pairwise=None rather than raising, so
    callers can render a "test failed" note instead of crashing the summary.
    """
    ranks = _mean_ranks(pivot)
    cols = list(pivot.columns)
    if len(cols) == 2:
        try:
            # scipy computes (and discards) a normal-approximation z-score
            # internally even when the exact distribution is used for the
            # returned p-value, so a 0 std-error (e.g. identical paired
            # samples) still warns despite the result being valid -- just
            # suppress it rather than treating it as a failure.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                stat, pval = wilcoxon(pivot[cols[0]], pivot[cols[1]])
        except Exception:
            return ranks, None, 'Wilcoxon signed-rank', float('nan'), float('nan')
        pairwise = pd.DataFrame([[1.0, pval], [pval, 1.0]], index=cols, columns=cols)
        return ranks, pairwise, 'Wilcoxon signed-rank', stat, pval

    try:
        stat, pval = friedmanchisquare(*[pivot[c].values for c in cols])
    except Exception:
        return ranks, None, 'Friedman', float('nan'), float('nan')
    pairwise = None
    if sp is not None:
        try:
            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.columns = cols
            nemenyi.index = cols
            pairwise = nemenyi
        except Exception:
            pairwise = None
    return ranks, pairwise, 'Friedman', stat, pval


def _pvalue_stars(p) -> str:
    """Significance-tier suffix for a p-value: '***' p<0.001, '**' p<0.01,
    '*' p<0.05, '' otherwise (including NaN)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _ranking_chain(ranks, pairwise) -> str:
    """Render a best->worst ranking chain, e.g. 'A > B ≈ C': adjacent levels
    join with '>' when pairwise-significant (p<0.05), else '≈' (not
    significantly different). Uses adjacent comparisons only (standard
    critical-difference convention) rather than every pairwise p-value, since
    significance is not transitive and a full-pairwise chain could misstate a
    strict total order. Falls back to '>' throughout (with a caveat) when no
    post-hoc pairwise matrix is available."""
    levels = list(ranks.index)
    if len(levels) < 2:
        return str(levels[0]) if levels else ''
    if pairwise is None:
        return " > ".join(str(l) for l in levels) + " (post-hoc unavailable; order by mean rank only)"
    parts = [str(levels[0])]
    for prev, cur in zip(levels, levels[1:]):
        sep = " > " if pairwise.loc[prev, cur] < 0.05 else " ≈ "
        parts.append(sep)
        parts.append(str(cur))
    return "".join(parts)


def _strata(df, cols):
    """Yield (label, sub_df) for each unique combination of `cols` present in
    df, so a test can be re-run separately per stratum instead of pooling
    across the other varying IVs. label is a human-readable 'k=v, k=v' string
    (using the bare column names). Yields a single ('', df) pair when cols is
    empty -- the no-stratification case."""
    if not cols:
        yield "", df
        return
    for vals, sub in df.groupby(cols, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        label = ", ".join(f"{c}={v}" for c, v in zip(cols, vals))
        yield label, sub


def _render_headline(df, iv_cols, non_ordinal_ivs, ordinal_iv_cols) -> str:
    lines = ["### Headline Results\n\n"]
    err_col = 'log_error_all'

    for key, col in ordinal_iv_cols:
        other_iv_cols = [c for k, c in iv_cols.items() if k != key]
        if other_iv_cols:
            lines.append(f"- `{key}`: correlation depends on other IVs -- "
                         "see per-stratum Spearman breakdown below.\n")
            continue
        try:
            rho, p = spearmanr(df[col].astype(float), df[err_col])
        except Exception as e:
            lines.append(f"- `{key}`: Spearman correlation failed ({e}).\n")
            continue
        if np.isnan(rho):
            lines.append(f"- `{key}`: not enough variation to compute a correlation.\n")
            continue
        direction = "higher values → lower error" if rho < 0 else "higher values → higher error"
        sig = "significant" if p < 0.05 else "not significant"
        lines.append(f"- `{key}`: {direction} (ρ={rho:.2f}, p={p:.3g}, {sig}).\n")

    for key in non_ordinal_ivs:
        col = iv_cols[key]
        other_iv_cols = [c for k, c in iv_cols.items() if k != key]
        if other_iv_cols:
            lines.append(f"- `{key}`: ranking depends on other IVs -- "
                         "see per-stratum Friedman/Nemenyi breakdown below.\n")
            continue
        pivot = _friedman_pivot(df, col, ['trial_idx'], err_col)
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            lines.append(f"- `{key}`: not enough complete blocks/levels to rank.\n")
            continue
        ranks, pairwise, _, _, _ = _rank_test(pivot)
        best = ranks.index[0]
        note = ""
        if pairwise is not None:
            worse = [str(lvl) for lvl in ranks.index[1:] if pairwise.loc[best, lvl] < 0.05]
            if worse:
                note = f", significantly better than {', '.join(worse)}"
            else:
                note = ", but not significantly better than any other level"
        lines.append(f"- `{key}`: best level is `{best}` (mean rank {ranks.iloc[0]:.2f}){note}.\n"
                     f"  Ranking: {_ranking_chain(ranks, pairwise)}\n")

    if len(lines) == 1:
        lines.append("_No ordinal or non-ordinal IVs with enough variation to summarize._\n")
    lines.append("\n")
    return "".join(lines)


_CAT_TERM = re.compile(r"C\(Q\('([^']+)'\)\)\[T\.(.+)\]$")
# Patsy also emits bracketed Treatment-coding terms without an explicit C()
# wrapper when the underlying column dtype is non-numeric (e.g. config_*
# columns reloaded from CSV as strings) -- same label treatment applies.
_BARE_CAT_TERM = re.compile(r"Q\('([^']+)'\)\[T\.(.+)\]$")
_NUM_TERM = re.compile(r"Q\('([^']+)'\)$")


def _prettify_term(term: str, col_to_key: dict) -> str:
    """Map a patsy term name (e.g. C(Q('config_train_cot'))[T.1.0]) back to a
    friendly IV label (e.g. cot=1.0), using the col->IV-key reverse map."""
    for pattern in (_CAT_TERM, _BARE_CAT_TERM):
        m = pattern.match(term)
        if m:
            col, level = m.groups()
            return f"{col_to_key.get(col, col)}={level}"
    m = _NUM_TERM.match(term)
    if m:
        col = m.group(1)
        return col_to_key.get(col, col)
    return term


def _render_lmm(df, iv_cols, non_ordinal_ivs, ordinal_iv_cols) -> str:
    out = ["### Linear Mixed Model (all IVs, All-Data log error)\n\n"]
    if smf is None:
        out.append("_statsmodels not installed; skipping LMM._\n\n")
        return "".join(out)
    if not iv_cols:
        out.append("_No IVs to fit._\n\n")
        return "".join(out)

    terms = []
    for key, col in iv_cols.items():
        terms.append(f"C(Q('{col}'))" if key in non_ordinal_ivs else f"Q('{col}')")
    formula = "log_error_all ~ " + " + ".join(terms)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, data=df, groups=df['trial_idx']).fit()
    except Exception as e:
        out.append(f"_LMM failed to fit: {e}_\n\n")
        return "".join(out)

    col_to_key = {col: key for key, col in iv_cols.items()}
    out.append("| Term | Coef | Std Err | p-value |\n|------|------|---------|--------|\n")
    for term, coef in fit.params.items():
        se = fit.bse.get(term, float('nan'))
        p = fit.pvalues.get(term, float('nan'))
        label = _prettify_term(term, col_to_key)
        out.append(f"| {label} | {coef:.4f} | {se:.4f} | {p:.3g} |\n")
    out.append("\n")
    return "".join(out)


def _render_friedman_nemenyi(df, err_col, non_ordinal_ivs, iv_cols) -> str:
    if not non_ordinal_ivs:
        return "_No non-ordinal IVs to rank._\n\n"
    out = ["**Friedman + Nemenyi (non-ordinal IVs; 2-level IVs fall back to a "
           "paired Wilcoxon signed-rank test, since Friedman's chi-square "
           "approximation is undefined below 3 treatments). Each IV is tested "
           "separately per stratum of the other varying IVs, rather than "
           "pooled across them, since pooling can mask interactions.**\n\n"]
    for key in non_ordinal_ivs:
        col = iv_cols[key]
        other_iv_cols = [c for k, c in iv_cols.items() if k != key]
        out.append(f"#### `{key}`\n\n")
        for stratum_label, sub_df in _strata(df, other_iv_cols):
            heading = f"  _{stratum_label}_\n\n" if stratum_label else ""
            pivot = _friedman_pivot(sub_df, col, ['trial_idx'], err_col)
            if pivot.shape[0] < 3 or pivot.shape[1] < 2:
                out.append(heading + f"  not enough complete blocks/levels for a rank test "
                            f"({pivot.shape[0]} blocks, {pivot.shape[1]} levels).\n\n")
                continue
            ranks, pairwise, test_name, stat, p = _rank_test(pivot)
            if np.isnan(p):
                out.append(heading + f"  {test_name} test failed to compute.\n\n")
                continue
            out.append(heading + f"  {test_name} statistic={stat:.3f}, p={p:.3g} (n={pivot.shape[0]} blocks)\n\n")
            out.append("  | Level | Mean Rank |\n  |-------|-----------|\n")
            for lvl, r in ranks.items():
                out.append(f"  | {lvl} | {r:.2f} |\n")
            out.append("\n")
            if pairwise is None:
                out.append("  _Post-hoc pairwise test unavailable (scikit-posthocs not installed, or it failed)._\n\n")
                continue
            out.append("  Pairwise post-hoc p-values (`*` p<0.05, `**` p<0.01, `***` p<0.001):\n\n")
            out.append("  | | " + " | ".join(str(c) for c in pairwise.columns) + " |\n")
            out.append("  |---" * (len(pairwise.columns) + 1) + "|\n")
            for lvl in pairwise.index:
                cells = [f"{pairwise.loc[lvl, c]:.3g}{_pvalue_stars(pairwise.loc[lvl, c])}"
                         for c in pairwise.columns]
                out.append(f"  | {lvl} | " + " | ".join(cells) + " |\n")
            out.append("\n")
    return "".join(out)


def _render_spearman(df, err_col, ordinal_iv_cols, iv_cols) -> str:
    if not ordinal_iv_cols:
        return "_No ordinal IVs to correlate._\n\n"
    out = ["**Spearman correlation (ordinal IVs). Each IV is correlated "
           "separately per stratum of the other varying IVs, rather than "
           "pooled across them.**\n\n"]
    for key, col in ordinal_iv_cols:
        other_iv_cols = [c for k, c in iv_cols.items() if k != key]
        out.append(f"#### `{key}`\n\n")
        out.append("| Stratum | rho | p-value |\n|---------|-----|--------|\n")
        for stratum_label, sub_df in _strata(df, other_iv_cols):
            label = stratum_label or "(all)"
            try:
                rho, p = spearmanr(sub_df[col].astype(float), sub_df[err_col])
            except Exception as e:
                out.append(f"| {label} | failed ({e}) | - |\n")
                continue
            if np.isnan(rho):
                out.append(f"| {label} | not enough variation | - |\n")
                continue
            out.append(f"| {label} | {rho:.3f} | {p:.3g} |\n")
        out.append("\n")
    return "".join(out)
