"""
Create figures and tables for paper
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import wandb

def plot_log_proxy_vs_step(df, all_runs_data, model_name):
    """Generates combined and individual L2 proxy vs step plots."""

    def get_strategy_color(strategy):
        s = strategy.lower()
        if 'fp64' in s: return '#1F77B4'  # Blue
        if 'fp32' in s: return '#7F7F7F'  # Gray
        if 'dynamic' in s or 'curvature' in s: return '#FF7F0E'  # Orange
        return '#2CA02C'

    ordered_equations = ['Convection', 'Reaction', 'Wave', 'Allen']

    print("\n--- Generating combined Log Proxy vs Step plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    axes = axes.flatten()
    legend_handles = {}

    for i, equation in enumerate(ordered_equations):
        ax = axes[i]
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        if not eq_runs:
            ax.set_title(f'{equation}\n(No data)', fontsize=22)
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=20)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            continue

        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))
        # Just grab the active dynamic strategy, ignoring the hardcoded proxy text
        dyn_strat = next((s for s in all_strategies_eq if 'dynamic' in s.lower() or 'curvature' in s.lower()), None)
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)
        strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

        for strategy in strategies_to_plot:
            strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
            if not strat_runs: continue
            all_histories = []
            for run_data in strat_runs:
                history = run_data['wandb_run'].history()
                if '_step' in history.columns and 'smoothed_log_proxy' in history.columns:
                    h = history[['_step', 'smoothed_log_proxy']].dropna()
                    h = h[h['smoothed_log_proxy'] != 0.0]
                    if not h.empty:
                        all_histories.append(h.sort_values('_step').reset_index(drop=True))
            if not all_histories: continue
            min_len = min(len(h) for h in all_histories)
            if min_len == 0: continue
            aligned_steps = np.array([h['_step'].values[:min_len] for h in all_histories])
            aligned_proxy = np.array([h['smoothed_log_proxy'].values[:min_len] for h in all_histories])
            avg_steps, mean_proxy, std_proxy = np.mean(aligned_steps, axis=0), np.mean(aligned_proxy, axis=0), np.std(
                aligned_proxy, axis=0)
            color = get_strategy_color(strategy)

            if 'dynamic' in strategy.lower():
                label = 'Dynamic'
            else:
                label = strategy

            line, = ax.plot(avg_steps, mean_proxy, color=color, label=label, linewidth=2.5)
            ax.fill_between(avg_steps, mean_proxy - std_proxy, mean_proxy + std_proxy, color=color, alpha=0.2)
            if label not in legend_handles:
                legend_handles[label] = line

        ax.set_title(equation, fontsize=22)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i >= 2: ax.set_xlabel('Training Step', fontsize=20)
        if i % 2 == 0: ax.set_ylabel(r'$\tilde{z}_t$')

    sorted_handles = dict(sorted(legend_handles.items(), key=lambda item: (
        'fp64' not in item[0].lower(),
        'fp32' not in item[0].lower(),
        'dynamic' not in item[0].lower()
    )))
    fig.legend(sorted_handles.values(), sorted_handles.keys(), loc='upper right', fontsize=21, title='Strategy',
               title_fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    filename = f'log_proxy_vs_step_combined_{model_name}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved combined plot to {filename}")
    plt.close(fig)

    print("--- Generating individual Log Proxy vs Step plots ---")
    all_strategies_global = sorted(set(r['Strategy'] for r in all_runs_data))
    dynamic_strats_global = [s for s in all_strategies_global if 'dynamic' in s.lower() or 'curvature' in s.lower()]

    for equation in df['Equation'].unique():
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        if not eq_runs: continue
        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)

        for dyn_strat in dynamic_strats_global:
            if dyn_strat not in all_strategies_eq: continue

            fig, ax = plt.subplots(figsize=(10, 6))
            strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

            for strategy in strategies_to_plot:
                strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
                if not strat_runs: continue
                all_histories = []
                for run_data in strat_runs:
                    history = run_data['wandb_run'].history()
                    if '_step' in history.columns and 'smoothed_log_proxy' in history.columns:
                        h = history[['_step', 'smoothed_log_proxy']].dropna()
                        h = h[h['smoothed_log_proxy'] != 0.0]
                        if not h.empty:
                            all_histories.append(h.sort_values('_step').reset_index(drop=True))
                if not all_histories: continue
                min_len = min(len(h) for h in all_histories)
                if min_len == 0: continue
                aligned_steps = np.array([h['_step'].values[:min_len] for h in all_histories])
                aligned_proxy = np.array([h['smoothed_log_proxy'].values[:min_len] for h in all_histories])
                avg_steps, mean_proxy, std_proxy = np.mean(aligned_steps, axis=0), np.mean(aligned_proxy,
                                                                                           axis=0), np.std(
                    aligned_proxy, axis=0)
                color = get_strategy_color(strategy)

                label = strategy

                ax.plot(avg_steps, mean_proxy, color=color, label=label, linewidth=2)
                ax.fill_between(avg_steps, mean_proxy - std_proxy, mean_proxy + std_proxy, color=color, alpha=0.2)

            ax.set_xlabel('Training Step', fontsize=20)
            ax.set_ylabel(r'$\tilde{z}_t$', fontsize=20)
            ax.set_ylim(1.5, 4.0)  # Ensure y-axis covers the baseline properly
            # ax.set_title(f'Log Proxy vs Step - {equation}', fontsize=21)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Strategy', fontsize=21)
            plt.tight_layout()

            proxy_match = re.search(r'\(Proxy (.*?)\)', dyn_strat)
            proxy_text = f"_proxy_{proxy_match.group(1).replace('.', '_')}" if proxy_match else "_dynamic"
            filename = f'log_proxy_vs_step_{equation.lower().replace(" ", "_")}_{model_name}{proxy_text}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_slope_proxy_vs_step(df, all_runs_data, model_name):
    """For each dynamic strategy, creates separate proxy slope vs step plots."""

    def get_strategy_color(strategy):
        s = strategy.lower()
        if 'fp64' in s: return '#1F77B4'
        if 'fp32' in s: return '#7F7F7F'
        if 'dynamic' in s or 'curvature' in s: return '#FF7F0E'
        return '#2CA02C'

    equations = df['Equation'].unique()

    for equation in equations:
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        all_strategies = sorted(set(r['Strategy'] for r in eq_runs))

        fp32_strat = next((s for s in all_strategies if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)
        dynamic_strats = [s for s in all_strategies if 'dynamic' in s.lower() or 'curvature' in s.lower()]

        if not dynamic_strats:
            continue

        for dyn_strat in dynamic_strats:
            fig, ax = plt.subplots(figsize=(10, 6))
            strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

            for strategy in strategies_to_plot:
                strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
                if not strat_runs: continue

                all_histories = []
                for run_data in strat_runs:
                    history = run_data['wandb_run'].history()
                    if '_step' in history.columns and 'proxy_slope' in history.columns:
                        h = history[['_step', 'proxy_slope']].dropna()
                        h = h[h['proxy_slope'] != 0.0]
                        if not h.empty:
                            all_histories.append(h.sort_values('_step').reset_index(drop=True))

                if len(all_histories) < 1: continue
                min_len = min(len(h) for h in all_histories)
                if min_len == 0: continue

                aligned_data = np.array([h['proxy_slope'].values[:min_len] for h in all_histories])
                steps = all_histories[0]['_step'].values[:min_len]
                mean, std = np.mean(aligned_data, axis=0), np.std(aligned_data, axis=0)
                color = get_strategy_color(strategy)
                ax.plot(steps, mean, color=color, label=strategy, linewidth=2)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

            ax.set_xlabel('Training Step', fontsize=22);
            ax.set_ylabel('Proxy Slope', fontsize=22)
            ax.set_title(f'Proxy Slope vs Step - {equation}', fontsize=21)

            ax.grid(True, alpha=0.3);
            ax.legend(title='Strategy', fontsize=21)
            plt.tight_layout()

            proxy_match = re.search(r'\(Proxy (.*?)\)', dyn_strat)
            proxy_text = f"_proxy_{proxy_match.group(1).replace('.', '_')}" if proxy_match else "_dynamic"
            filename = f'slope_proxy_vs_step_{equation.lower().replace(" ", "_")}_{model_name}{proxy_text}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_l2_vs_walltime(df, all_runs_data, model_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    MIN_ACTIVE_RUNS = 1
    GRID_POINTS = 300

    def get_strategy_color(strategy):
        s = strategy.lower()
        if 'fp64' in s: return '#1F77B4'  # Blue
        if 'fp32' in s: return '#7F7F7F'  # Gray
        if 'dynamic' in s or 'curvature' in s: return '#FF7F0E'  # Orange
        return '#2CA02C'

    def _prepare_history(history_df):
        if '_runtime' not in history_df.columns or 'rRMSE' not in history_df.columns:
            return None
        h = history_df[['_runtime', 'rRMSE']].dropna().copy()
        if h.empty: return None
        h = h.sort_values('_runtime').reset_index(drop=True)
        h = h.groupby('_runtime', as_index=False).last()
        if len(h) < 2: return None
        t = h['_runtime'].to_numpy(dtype=float)
        y = h['rRMSE'].to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]
        if len(t) < 2: return None
        return t, y

    def _aggregate_strategy_curves(strat_runs):
        prepared = []
        for run_data in strat_runs:
            history = run_data['wandb_run'].history()
            out = _prepare_history(history)
            if out is not None:
                prepared.append(out)

        if not prepared: return None
        max_time = max(t[-1] for t, _ in prepared)
        if not np.isfinite(max_time) or max_time <= 0: return None

        grid = np.linspace(0.0, max_time, GRID_POINTS)
        interpolated = []
        for t, y in prepared:
            vals = np.full_like(grid, np.nan, dtype=float)
            valid = grid <= t[-1]
            vals[valid] = np.interp(grid[valid], t, y)
            interpolated.append(vals)

        interpolated = np.vstack(interpolated)
        active_counts = np.sum(~np.isnan(interpolated), axis=0)
        valid_cols = active_counts >= MIN_ACTIVE_RUNS
        if not np.any(valid_cols):
            valid_cols = active_counts >= 1
            if not np.any(valid_cols): return None

        grid = grid[valid_cols]
        vals = interpolated[:, valid_cols]
        return grid, np.nanmean(vals, axis=0), np.nanstd(vals, axis=0)

    ordered_equations = ['Convection', 'Reaction', 'Wave', 'Allen']

    print("\n--- Generating combined rRMSE vs Wall Time plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    axes = axes.flatten()
    legend_handles = {}

    for i, equation in enumerate(ordered_equations):
        ax = axes[i]
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]

        if not eq_runs:
            ax.set_title(f'{equation}\n(No data)', fontsize=22)
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=20)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            continue

        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))
        # Just grab the active dynamic strategy, ignoring the hardcoded proxy text
        dyn_strat = next((s for s in all_strategies_eq if 'dynamic' in s.lower() or 'curvature' in s.lower()), None)
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)
        strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

        for strategy in strategies_to_plot:
            strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
            if not strat_runs: continue

            agg = _aggregate_strategy_curves(strat_runs)
            if agg is None: continue

            avg_runtime, mean_rrmse, std_rrmse = agg
            color = get_strategy_color(strategy)

            label = 'Dynamic' if 'dynamic' in strategy.lower() else strategy
            line, = ax.plot(avg_runtime, mean_rrmse, color=color, label=label, linewidth=2.5)
            ax.fill_between(avg_runtime, mean_rrmse - std_rrmse, mean_rrmse + std_rrmse,
                            color=color, alpha=0.2)

            if label not in legend_handles:
                legend_handles[label] = line

        ax.set_title(equation, fontsize=22)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i >= 2: ax.set_xlabel('Wall Time (s)', fontsize=20)
        if i % 2 == 0: ax.set_ylabel("rRMSE", fontsize=20)

    sorted_handles = dict(sorted(
        legend_handles.items(),
        key=lambda item: ('fp64' not in item[0].lower(), 'fp32' not in item[0].lower(), 'dynamic' not in item[0].lower())
    ))
    fig.legend(sorted_handles.values(), sorted_handles.keys(),
               loc='upper right', fontsize=21, title='Strategy', title_fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    filename = f'rRMSE_vs_walltime_combined_{model_name}.svg'
    plt.savefig(filename, dpi=300)
    print(f"Saved combined plot to {filename}")
    plt.close(fig)

    print("--- Generating individual rRMSE vs Wall Time plots ---")
    all_strategies_global = sorted(set(r['Strategy'] for r in all_runs_data))
    dynamic_strats_global = [s for s in all_strategies_global if 'dynamic' in s.lower() or 'curvature' in s.lower()]

    for equation in df['Equation'].unique():
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        if not eq_runs: continue

        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)

        for dyn_strat in dynamic_strats_global:
            if dyn_strat not in all_strategies_eq: continue

            fig, ax = plt.subplots(figsize=(10, 6))
            strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

            for strategy in strategies_to_plot:
                strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
                if not strat_runs: continue

                agg = _aggregate_strategy_curves(strat_runs)
                if agg is None: continue

                avg_runtime, mean_rrmse, std_rrmse = agg
                color = get_strategy_color(strategy)

                ax.plot(avg_runtime, mean_rrmse, color=color, label=strategy, linewidth=2)
                ax.fill_between(avg_runtime, mean_rrmse - std_rrmse, mean_rrmse + std_rrmse,
                                color=color, alpha=0.2)

            ax.set_xlabel('Wall Time (s)', fontsize=20)
            ax.set_ylabel("rRMSE", fontsize=20)
            ax.set_title(f'rRMSE vs. Wall Time - {equation}', fontsize=21)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Strategy', fontsize=21)
            plt.tight_layout()

            proxy_match = re.search(r'\(Proxy (.*?)\)', dyn_strat)
            proxy_text = f"_proxy_{proxy_match.group(1).replace('.', '_')}" if proxy_match else "_dynamic"
            filename = f'rRMSE_vs_walltime_{equation.lower().replace(" ", "_")}_{model_name}{proxy_text}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_l2_vs_step(df, all_runs_data, model_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    def get_strategy_color(strategy):
        s = strategy.lower()
        if 'fp64' in s: return '#1F77B4'
        if 'fp32' in s: return '#7F7F7F'
        if 'dynamic' in s or 'curvature' in s: return '#FF7F0E'
        return '#2CA02C'

    ordered_equations = ['Convection', 'Reaction', 'Wave', 'Allen']

    print("\n--- Generating combined rRMSE vs Step plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    axes = axes.flatten()
    legend_handles = {}

    for i, equation in enumerate(ordered_equations):
        ax = axes[i]
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        if not eq_runs:
            ax.set_title(f'{equation}\n(No data)', fontsize=22)
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=20)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            continue

        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))

        dyn_strat = next((s for s in all_strategies_eq if 'dynamic' in s.lower() or 'curvature' in s.lower()), None)
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)
        strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

        for strategy in strategies_to_plot:
            strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
            if not strat_runs: continue
            all_histories = []
            for run_data in strat_runs:
                history = run_data['wandb_run'].history()
                if '_step' in history.columns and 'rRMSE' in history.columns:
                    h = history[['_step', 'rRMSE']].dropna()
                    all_histories.append(h.sort_values('_step').reset_index(drop=True))
            if not all_histories: continue
            min_len = min(len(h) for h in all_histories)
            if min_len == 0: continue
            aligned_steps = np.array([h['_step'].values[:min_len] for h in all_histories])
            aligned_rrmse = np.array([h['rRMSE'].values[:min_len] for h in all_histories])
            avg_steps, mean_rrmse, std_rrmse = np.mean(aligned_steps, axis=0), np.mean(aligned_rrmse, axis=0), np.std(
                aligned_rrmse, axis=0)
            color = get_strategy_color(strategy)

            label = 'Dynamic' if 'dynamic' in strategy.lower() else strategy
            line, = ax.plot(avg_steps, mean_rrmse, color=color, label=label, linewidth=2.5)
            ax.fill_between(avg_steps, mean_rrmse - std_rrmse, mean_rrmse + std_rrmse, color=color, alpha=0.2)
            if label not in legend_handles: legend_handles[label] = line

        ax.set_title(equation, fontsize=22)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i >= 2: ax.set_xlabel('Training Step', fontsize=20)
        if i % 2 == 0: ax.set_ylabel("rRMSE", fontsize=20)

    sorted_handles = dict(sorted(
        legend_handles.items(),
        key=lambda item: ('fp64' not in item[0].lower(), 'fp32' not in item[0].lower(), 'dynamic' not in item[0].lower())
    ))
    fig.legend(sorted_handles.values(), sorted_handles.keys(), loc='upper right', fontsize=21, title='Strategy',
               title_fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    filename = f'l2_vs_step_combined_{model_name}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved combined plot to {filename}")
    plt.close(fig)

    print("--- Generating individual rRMSE vs Step plots ---")
    all_strategies_global = sorted(set(r['Strategy'] for r in all_runs_data))
    dynamic_strats_global = [s for s in all_strategies_global if 'dynamic' in s.lower() or 'curvature' in s.lower()]

    for equation in df['Equation'].unique():
        eq_runs = [r for r in all_runs_data if r['Equation'] == equation]
        if not eq_runs: continue
        all_strategies_eq = sorted(set(r['Strategy'] for r in eq_runs))
        fp32_strat = next((s for s in all_strategies_eq if 'fp32' in s.lower() and 'dynamic' not in s.lower()), None)
        fp64_strat = next((s for s in all_strategies_eq if 'fp64' in s.lower() and 'dynamic' not in s.lower()), None)

        for dyn_strat in dynamic_strats_global:
            if dyn_strat not in all_strategies_eq: continue

            fig, ax = plt.subplots(figsize=(10, 6))
            strategies_to_plot = [s for s in [fp32_strat, fp64_strat, dyn_strat] if s is not None]

            for strategy in strategies_to_plot:
                strat_runs = [r for r in eq_runs if r['Strategy'] == strategy]
                if not strat_runs: continue
                all_histories = []
                for run_data in strat_runs:
                    history = run_data['wandb_run'].history()
                    if '_step' in history.columns and 'rRMSE' in history.columns:
                        h = history[['_step', 'rRMSE']].dropna()
                        all_histories.append(h.sort_values('_step').reset_index(drop=True))
                if not all_histories: continue
                min_len = min(len(h) for h in all_histories)
                if min_len == 0: continue
                aligned_steps = np.array([h['_step'].values[:min_len] for h in all_histories])
                aligned_rrmse = np.array([h['rRMSE'].values[:min_len] for h in all_histories])
                avg_steps, mean_rrmse, std_rrmse = np.mean(aligned_steps, axis=0), np.mean(aligned_rrmse,
                                                                                           axis=0), np.std(
                    aligned_rrmse, axis=0)
                color = get_strategy_color(strategy)

                ax.plot(avg_steps, mean_rrmse, color=color, label=strategy, linewidth=2)
                ax.fill_between(avg_steps, mean_rrmse - std_rrmse, mean_rrmse + std_rrmse, color=color, alpha=0.2)

            ax.set_xlabel('Training Step', fontsize=20)
            ax.set_ylabel('rRMSE', fontsize=20)
            ax.set_title(f'rRMSE vs. Step - {equation}', fontsize=21)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Strategy', fontsize=21)
            plt.tight_layout()
            proxy_match = re.search(r'\(Proxy (.*?)\)', dyn_strat)
            proxy_text = f"_proxy_{proxy_match.group(1).replace('.', '_')}" if proxy_match else "_dynamic"
            filename = f'l2_vs_step_{equation.lower().replace(" ", "_")}_{model_name}{proxy_text}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_single_seed_from_project(
        project,
        seed,
        entity="lokious-wageningen-uinversity",
        out_prefix="single_seed_comparison"
):
    """
    Download single-seed runs example from a W&B project and plot (Figure 9):
      1) rRMSE vs step
      2) precision_fp64 vs step
      3) smoothed_log_proxy vs step
    """
    import wandb
    import matplotlib.pyplot as plt

    def _get_strategy_color(strategy):
        s = strategy.lower()
        if 'fp64' in s: return '#1F77B4'
        if 'fp32' in s: return '#7F7F7F'
        if 'dynamic' in s: return '#FF7F0E'
        return '#2CA02C'

    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}")

    selected = {"fp32": None, "fp64": None, "dynamic": None}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config or {}
        if cfg.get("seed") != seed:
            continue

        raw_strategy = cfg.get("strategy", "N/A")
        if "switching" in raw_strategy.lower():
            continue

        if raw_strategy == "fp64_curvature":
            raw_strategy = "fp64"
        elif raw_strategy == "fp32_curvature":
            raw_strategy = "fp32"

        strategy = raw_strategy.replace("_", " ")
        is_dynamic = ("dynamic" in strategy.lower()) or ("curvature" in strategy.lower())

        if not is_dynamic and "fp32" in strategy.lower():
            selected["fp32"] = run
        elif not is_dynamic and "fp64" in strategy.lower():
            selected["fp64"] = run
        elif is_dynamic:
            if selected["dynamic"] is None:
                selected["dynamic"] = run

    selected = {k: v for k, v in selected.items() if v is not None}
    if not selected:
        print(f"No runs found for project={project}, seed={seed}")
        return

    histories = {}
    for key, run in selected.items():
        h = run.history()
        if "_step" not in h.columns:
            continue
        cols = ["_step", "rRMSE", "precision_fp64", "smoothed_log_proxy"]
        cols = [c for c in cols if c in h.columns]
        h = h[cols].dropna(subset=["_step"]).sort_values("_step").reset_index(drop=True)
        histories[key] = h

    if not histories:
        print("No valid histories found with _step.")
        return

    common_steps = None
    for h in histories.values():
        steps = set(h["_step"].values.tolist())
        common_steps = steps if common_steps is None else common_steps & steps

    if not common_steps:
        print("No common steps across strategies to align.")
        return

    common_steps = sorted(common_steps)

    def _align_metric(h, metric):
        if metric not in h.columns:
            return None
        h2 = h[h["_step"].isin(common_steps)]
        return h2.set_index("_step").loc[common_steps][metric].values

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    metrics = [
        ("rRMSE", "rRMSE"),
        ("precision_fp64", "precision fp64"),
        ("smoothed_log_proxy", "log curvature proxy"),
    ]

    legend_handles = {}

    for ax, (metric_key, y_label) in zip(axes, metrics):
        for key, h in histories.items():
            y = _align_metric(h, metric_key)
            if y is None:
                continue
            label = key.upper() if key in ["fp32", "fp64"] else "Dynamic"
            color = _get_strategy_color(label)
            line, = ax.plot(common_steps, y, color=color, linewidth=2, label=label)
            if label not in legend_handles:
                legend_handles[label] = line

        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training Step")

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="upper right",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Strategy"
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    safe_project = project.replace("/", "_")
    filename = f"{out_prefix}_{safe_project}_seed{seed}.svg"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filename}")


def plot_training_time_vs_accuracy(df, model_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    print(f"\n--- Normalized Speed vs Accuracy Combined Scatter Plot ({model_name}) ---")

    mask_fp64 = df['Strategy'].str.lower().str.contains('fp64')
    mask_dynamic = df['Strategy'].str.lower().str.contains('dynamic|curvature')
    plot_df = df[mask_fp64 | mask_dynamic].copy()

    if plot_df.empty:
        print("No FP64 or Dynamic data to plot.")
        return

    # Added Irradiance
    ordered_equations = ['Convection', 'Reaction', 'Wave', 'Allen', 'Irradiance']
    present_equations = [eq for eq in ordered_equations if eq in plot_df['Equation'].unique()]

    best_strats = []
    for eq in present_equations:
        eq_df = plot_df[plot_df['Equation'] == eq]
        fp64_strats = eq_df[eq_df['Strategy'].str.lower().str.contains('fp64')]['Strategy'].unique()
        best_strats.extend(fp64_strats)

        dyn_df = eq_df[eq_df['Strategy'].str.lower().str.contains('dynamic|curvature')]
        if not dyn_df.empty:
            best_dyn = dyn_df.groupby('Strategy')['Training Time (s)'].mean().idxmin()
            best_strats.append(best_dyn)

    plot_df = plot_df[plot_df['Strategy'].isin(best_strats)].copy()

    present_base_strats = set()
    for strat in plot_df['Strategy'].unique():
        s_lower = strat.lower()
        if 'dynamic' in s_lower or 'curvature' in s_lower:
            present_base_strats.add('dynamic')
        else:
            present_base_strats.add('fp64')

    plot_df['Normalized Time (%)'] = np.nan
    for eq in present_equations:
        eq_df = plot_df[plot_df['Equation'] == eq]
        fp64_df = eq_df[eq_df['Strategy'].str.lower() == 'fp64']
        if fp64_df.empty:
            fp64_df = eq_df[eq_df['Strategy'].str.lower().str.contains('fp64')]

        if not fp64_df.empty:
            mean_fp64_time = fp64_df['Training Time (s)'].mean()
            plot_df.loc[plot_df['Equation'] == eq, 'Normalized Time (%)'] = (eq_df[
                                                                                 'Training Time (s)'] / mean_fp64_time) * 100

    plot_df.dropna(subset=['Normalized Time (%)'], inplace=True)

    print("\n  -> EXACT configurations plotted in this figure:")
    for eq in present_equations:
        print(f"     [{eq}]")
        eq_df = plot_df[plot_df['Equation'] == eq]
        for strat in eq_df['Strategy'].unique():
            hp_val = eq_df[eq_df['Strategy'] == strat]['Hyperparams'].iloc[0]
            hp_str = hp_val.replace("(", "").replace(")", "").replace("'", "")
            print(f"       * {strat}")
            print(f"         Hyperparams: {hp_str}")
    print("  " + "-" * 60)

    colors = {
        'fp64': '#1F77B4',
        'dynamic': '#FF7F0E'
    }

    # Added Irradiance shape
    markers = {
        'Convection': 'o',
        'Reaction': 's',
        'Wave': '^',
        'Allen': 'D',
        'Irradiance': 'P'
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    for equation in present_equations:
        eq_df = plot_df[plot_df['Equation'] == equation]

        for strategy in sorted(eq_df['Strategy'].unique()):
            strat_df = eq_df[eq_df['Strategy'] == strategy]

            s_lower = strategy.lower()
            if 'dynamic' in s_lower or 'curvature' in s_lower:
                base_strat = 'dynamic'
            else:
                base_strat = 'fp64'

            c = colors[base_strat]
            m = markers.get(equation, 'x')

            ax.scatter(strat_df['Normalized Time (%)'], strat_df['rMAE'],
                       c=c, marker=m, s=150, alpha=0.6,
                       edgecolors='black', linewidth=1.5)

            ax.scatter(strat_df['Normalized Time (%)'].mean(), strat_df['rMAE'].mean(),
                       c=c, marker=m, s=400, alpha=1.0,
                       edgecolors='black', linewidth=2.5, zorder=5)

        time_vals = eq_df['Normalized Time (%)'].values
        rmae_vals = eq_df['rMAE'].values

        pareto_mask = np.ones(len(eq_df), dtype=bool)
        for i in range(len(eq_df)):
            for j in range(len(eq_df)):
                if i != j and time_vals[j] < time_vals[i] and rmae_vals[j] < rmae_vals[i]:
                    pareto_mask[i] = False
                    break

        pareto_df = eq_df[pareto_mask]
        if len(pareto_df) > 1:
            pareto_sorted = pareto_df.sort_values('Normalized Time (%)')
            ax.plot(pareto_sorted['Normalized Time (%)'], pareto_sorted['rMAE'],
                    linestyle='--', color='black', alpha=0.3, linewidth=2.0, zorder=1)

    ax.axvline(x=100, color='red', linestyle=':', alpha=0.6, linewidth=2.5, zorder=0)

    legend_elements = [
        Line2D([0], [0], color='w', label='--- Strategy ---'),
    ]

    if 'fp64' in present_base_strats:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='FP64',
                                      markerfacecolor=colors['fp64'], markersize=14, markeredgecolor='black'))
    if 'dynamic' in present_base_strats:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Dynamic',
                                      markerfacecolor=colors['dynamic'], markersize=14, markeredgecolor='black'))

    legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=2.5, label='FP64 Benchmark (100%)'))
    legend_elements.append(Line2D([0], [0], color='w', label=' '))
    legend_elements.append(Line2D([0], [0], color='w', label='--- Equation ---'))

    for eq in present_equations:
        legend_elements.append(
            Line2D([0], [0], marker=markers[eq], color='w', label=eq,
                   markerfacecolor='gray', markersize=14, markeredgecolor='black')
        )

    leg = ax.legend(handles=legend_elements, fontsize=13, loc='upper left',
                    framealpha=0.95, bbox_to_anchor=(1.02, 1.0))

    for text in leg.get_texts():
        if text.get_text() in ['--- Strategy ---', '--- Equation ---']:
            text.set_fontweight('bold')
            text.set_fontsize(14)

    ax.set_xlim(0, 150)
    ax.set_xlabel('Relative Training Time (% of FP64 Mean)', fontsize=21, fontweight='bold')
    ax.set_ylabel('rMAE (Error)', fontsize=21, fontweight='bold')
    ax.set_title('Irrradiance',#f'Normalized Speed vs Accuracy Overview ({model_name})'
                 fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    filename = 'Irradiance'#f'speed_vs_accuracy_combined_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved combined plot: {filename}")
    plt.close(fig)

    print(f"--- Normalized Speed vs Accuracy Subplots ({model_name}) ---")

    fig_sub, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    for i, equation in enumerate(ordered_equations):
        if i >= 4: break

        ax_sub = axes[i]

        if equation not in present_equations:
            ax_sub.set_title(f'{equation}\n(No data)', fontsize=20, fontweight='bold')
            ax_sub.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                        transform=ax_sub.transAxes, fontsize=20)
            ax_sub.grid(True, which="both", ls="--", alpha=0.4)
            ax_sub.set_xlim(0, 150)
            ax_sub.set_ylim(0, 0.04)
            continue

        eq_df = plot_df[plot_df['Equation'] == equation]
        m = markers.get(equation, 'o')

        for strategy in sorted(eq_df['Strategy'].unique()):
            strat_df = eq_df[eq_df['Strategy'] == strategy]

            s_lower = strategy.lower()
            if 'dynamic' in s_lower or 'curvature' in s_lower:
                base_strat = 'dynamic'
            else:
                base_strat = 'fp64'

            c = colors[base_strat]

            ax_sub.scatter(strat_df['Normalized Time (%)'], strat_df['rMAE'],
                           c=c, marker=m, s=150, alpha=0.6,
                           edgecolors='black', linewidth=1.5)

            ax_sub.scatter(strat_df['Normalized Time (%)'].mean(), strat_df['rMAE'].mean(),
                           c=c, marker=m, s=400, alpha=1.0,
                           edgecolors='black', linewidth=2.5, zorder=5)

        time_vals = eq_df['Normalized Time (%)'].values
        rmae_vals = eq_df['rMAE'].values

        pareto_mask = np.ones(len(eq_df), dtype=bool)
        for ii in range(len(eq_df)):
            for jj in range(len(eq_df)):
                if ii != jj and time_vals[jj] < time_vals[ii] and rmae_vals[jj] < rmae_vals[ii]:
                    pareto_mask[ii] = False
                    break

        pareto_df = eq_df[pareto_mask]
        if len(pareto_df) > 1:
            pareto_sorted = pareto_df.sort_values('Normalized Time (%)')
            ax_sub.plot(pareto_sorted['Normalized Time (%)'], pareto_sorted['rMAE'],
                        linestyle='--', color='black', alpha=0.3, linewidth=2.0, zorder=1)

        ax_sub.axvline(x=100, color='red', linestyle=':', alpha=0.6, linewidth=2.5, zorder=0)
        ax_sub.set_title(equation, fontsize=20, fontweight='bold')
        ax_sub.grid(True, alpha=0.3, linestyle='--')
        ax_sub.set_xlim(0, 150)
        ax_sub.set_ylim(0, 0.04)

        if i >= 2:
            ax_sub.set_xlabel('Relative Training Time (%)', fontsize=20, fontweight='bold')
        if i % 2 == 0:
            ax_sub.set_ylabel('rMAE (Error)', fontsize=20, fontweight='bold')

    sub_legend_elements = []

    if 'fp64' in present_base_strats:
        sub_legend_elements.append(Line2D([0], [0], marker='o', color='w', label='FP64',
                                          markerfacecolor=colors['fp64'], markersize=14, markeredgecolor='black'))
    if 'dynamic' in present_base_strats:
        sub_legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Dynamic',
                                          markerfacecolor=colors['dynamic'], markersize=14, markeredgecolor='black'))

    sub_legend_elements.append(
        Line2D([0], [0], color='red', linestyle=':', linewidth=2.5, label='FP64 Benchmark (100%)'))

    fig_sub.legend(handles=sub_legend_elements, fontsize=21, loc='upper center',
                   ncol=len(sub_legend_elements), bbox_to_anchor=(0.5, 1.05), framealpha=0.95)

    fig_sub.subplots_adjust(top=0.90)

    filename_sub = f'speed_vs_accuracy_subplots_{model_name}.png'
    plt.savefig(filename_sub, dpi=300, bbox_inches='tight')
    print(f"  Saved 2x2 subplots: {filename_sub}")
    plt.close(fig_sub)


def generate_results_summary(group=None):

    api = wandb.Api()
    group_col = str(group) if group is not None else None

    def safe_tag(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

    def get_group_value(config, summary, run, project, group_name):
        if group_name is None:
            return None

        group_key = str(group_name).lower()
        for source in (config or {}, summary or {}):
            for key, value in source.items():
                if str(key).lower() == group_key:
                    return value

        if group_key == "diagnostic_interval":
            texts = [project, getattr(run, "name", ""), getattr(run, "display_name", "")]
            patterns = [
                r"diagnostic[_-]?interval[_=-]?(\d+)",
                r"interval[_-]?(\d+)",
                r"(\d+)[_-]?step",
            ]
            for text in texts:
                for pattern in patterns:
                    match = re.search(pattern, str(text), flags=re.IGNORECASE)
                    if match:
                        return int(match.group(1))

        return "N/A"

    ENTITY = "lokious-wageningen-uinversity"
    PROJECTS = [
        # "pinn_irradiance_dynamic_precision"
        # "pinn_wave_dynamic_precision_MLP",
        # "pinn_reaction_dynamic_precision_MLP",
        # "pinn_convection_dynamic_precision_MLP",
        # "pinn_allen_cahn_dynamic_precision_MLP",

        # "pinn_convection_dynamic_precision_PINNsFormer",
        # "pinn_reaction_dynamic_precision_PINNsFormer",
        # "pinn_convection_dynamic_precision_PINNMamba",
        # "pinn_wave_dynamic_precision_PINNMamba",
        # "pinn_wave_dynamic_precision_PINNsFormer",
        # "pinn_allen_cahn_dynamic_precision_PINNsFormer",

        # "pinn_wave_dynamic_precision_3layerMLP",
        # "pinn_reaction_dynamic_precision_3layerMLP",
        # "pinn_convection_dynamic_precision_3layerMLP",
        # "pinn_allen_cahn_dynamic_precision_3layerMLP",
        # "PINN_heat_dynamic_precision_MLP"
        # "pinn_convection_adaptive_weights"
        # "pinn_ns2dcg_dynamic_precisionPINN",
        # "pinn_ns2dc_dynamic_precision_PINN"
        # "pinn_convection_adaptive_weights"
        # "pinn_allen_cahn_paper_method_comparators"

        "pinn_convection_dynamic_precision_5step_MLP",
        "pinn_reaction_dynamic_precision_5step_MLP",
        "pinn_wave_dynamic_precision_5step_MLP",
        "pinn_allen_cahn_dynamic_precision_5step_MLP",
    ]

    print(f"Fetching runs from entity: '{ENTITY}'...")

    raw_runs_data = []

    for project in PROJECTS:
        runs = api.runs(path=f"{ENTITY}/{project}")
        print(f"Found {len(runs)} runs in project '{project}'")

        equation_name = project.split('_')[1].replace('-', ' ').title()
        model_name = project.split('_')[-1]
        if model_name == "3layer":
            model_name = "3layerMLP"
        elif model_name == "precision":
            model_name = "PINN"

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            config = run.config
            seed = config.get("seed")

            if seed is None or (seed>5):#
                continue

            if "training_time_seconds" not in run.summary or "rRMSE" not in run.summary or "rMAE" not in run.summary:
                continue

            raw_strategy = config.get("strategy", "N/A")
            if "switching" in raw_strategy.lower():
                continue

            run_data = {
                "Equation": equation_name,
                "Model": model_name,
                "Raw_Strategy": raw_strategy,
                "LOG_PROXY_LOW": config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None)),
                "Rescale_Derivative": config.get("rescale_derivative", False),
                "Training Time (s)": run.summary.get("training_time_seconds"),
                "rRMSE": run.summary.get("rRMSE"),
                "rMAE": run.summary.get("rMAE"),
                "Seed": seed,
                "wandb_run": run
            }
            if group_col is not None:
                run_data[group_col] = get_group_value(config, run.summary, run, project, group_col)
            raw_runs_data.append(run_data)

    if not raw_runs_data:
        print("No valid runs found with seeds between 0 and 4!")
        return

    df = pd.DataFrame(raw_runs_data)

    df["Training Time (s)"] = pd.to_numeric(df["Training Time (s)"], errors='coerce')
    df["rRMSE"] = pd.to_numeric(df["rRMSE"], errors='coerce')
    df["rMAE"] = pd.to_numeric(df["rMAE"], errors='coerce')
    df.dropna(subset=["Training Time (s)", "rRMSE", "rMAE"], inplace=True)
    if group_col is not None:
        df[group_col] = df[group_col].fillna("N/A")

    def clean_strategy(raw_strat):
        if pd.isna(raw_strat):
            return "N/A"
        raw_strat = str(raw_strat)

        if raw_strat == "fp64_curvature":
            return "fp64"
        elif raw_strat == "fp32_curvature":
            return "fp32"
        elif raw_strat in ["dynamic", "dynamic_precision_lbfgs", "dynamic precision lbfgs"]:
            return "dynamic precision"

        return raw_strat.replace('_', ' ')

    df['Strategy'] = df['Raw_Strategy'].apply(clean_strategy)

    mask_rescale = df['Rescale_Derivative'] == True
    df.loc[mask_rescale, 'Strategy'] = df.loc[mask_rescale, 'Strategy'] + " + Rescale"

    mask_needs_proxy = df['Strategy'].str.contains('dynamic', case=False, na=False)
    df.loc[mask_needs_proxy & pd.isna(df['LOG_PROXY_LOW']), 'LOG_PROXY_LOW'] = 2.0
    df['LOG_PROXY_LOW'] = df['LOG_PROXY_LOW'].fillna("N/A")
    df['Hyperparams'] = "LOG_PROXY_LOW: " + df['LOG_PROXY_LOW'].astype(str)

    # CONVERGENCE_THRESHOLD = 0.3
    CONVERGENCE_THRESHOLD = 1.5 #all result
    mask_diverged = df['rRMSE'] > CONVERGENCE_THRESHOLD

    if mask_diverged.any():
        diverged_runs = df[mask_diverged]
        print("\n" + "!" * 80)
        print(f"!!! WARNING: Dropping {len(diverged_runs)} run(s) that failed to converge (rRMSE > {CONVERGENCE_THRESHOLD}) !!!")
        for _, row in diverged_runs.iterrows():
            print(f"  -> Eq: {row['Equation']:<12} | Strategy: {row['Strategy']:<20} | Proxy: {str(row['LOG_PROXY_LOW']):<4} | Seed: {row['Seed']} | rRMSE = {row['rRMSE']:.4f}")
        print("!" * 80 + "\n")

        df = df[~mask_diverged].copy()
        if df.empty:
            print("No valid runs left after filtering for convergence! Exiting.")
            return

    all_runs_data = df.to_dict(orient='records')
    output_suffix = f"_by_{safe_tag(group_col)}" if group_col is not None else ""
    df.to_csv(f"all_runs_data{output_suffix}.csv", index=False)

    selection_keys = (["Equation", "Model", "Strategy"] if group_col is None
                      else [group_col, "Equation", "Model", "Strategy"])
    proxy_selection_keys = selection_keys + ["LOG_PROXY_LOW"]
    entity_keys = (["Equation", "Model"] if group_col is None
                   else [group_col, "Equation", "Model"])

    group_perf = df.groupby(proxy_selection_keys)['Training Time (s)'].mean().reset_index()
    print("Mean training times by group:")
    print(group_perf)

    best_idx = group_perf.groupby(selection_keys)['Training Time (s)'].idxmin()
    best_groups = group_perf.loc[best_idx, proxy_selection_keys]

    df = pd.merge(df, best_groups, on=proxy_selection_keys, how='inner')

    best_run_ids = set(df['wandb_run'].apply(lambda r: r.id))
    all_runs_data = [r for r in all_runs_data if r['wandb_run'].id in best_run_ids]

    print(f"\nFiltered down to {len(df)} optimal runs across {len(best_groups)} unique strategy groups.")

    print("\n" + "=" * 80)
    print("--- SELECTED OPTIMAL HYPERPARAMETERS (Used for Plots & Tables) ---")
    print("=" * 80)
    for group_key, best_group in best_groups.groupby(entity_keys):
        if group_col is None:
            eq, model = group_key
            print(f"\n[{model}] Equation: {eq}")
        else:
            group_value, eq, model = group_key
            print(f"\n[{model}] {group_col}: {group_value} | Equation: {eq}")
        for _, row in best_group.iterrows():
            print(f"  - Strategy: {row['Strategy']}")
            print(f"    LOG_PROXY_LOW: {row['LOG_PROXY_LOW']}")
    print("\n" + "=" * 80 + "\n")

    baseline_strats = {}
    for group_key, grouped_df in df.groupby(entity_keys):
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        eq_model_strats = grouped_df['Strategy'].unique()
        fp64_strats = [s for s in eq_model_strats if 'fp64' in s.lower() and 'dynamic' not in s.lower()]
        if fp64_strats:
            baseline_strats[key] = fp64_strats[0]

    agg_keys = (["Equation", "Model", "Strategy"] if group_col is None
                else [group_col, "Equation", "Model", "Strategy"])
    agg_df = df.groupby(agg_keys).agg(
        Mean_Time=('Training Time (s)', 'mean'), Std_Time=('Training Time (s)', 'std'),
        Mean_rRMSE=('rRMSE', 'mean'), Std_rRMSE=('rRMSE', 'std'),
        Mean_rMAE=('rMAE', 'mean'), Std_rMAE=('rMAE', 'std'),
        Run_Count=('Seed', 'count')
    ).reset_index()
    print(agg_df)

    agg_df = agg_df[agg_df['Run_Count'] >= 2]
    ordered_equations = ["Convection", "Reaction", "Wave", "Allen", "Irradiance"]
    agg_df["Equation"] = pd.Categorical(agg_df["Equation"], categories=ordered_equations, ordered=True)
    sort_cols = (["Equation", "Model", "Strategy"] if group_col is None
                 else [group_col, "Equation", "Model", "Strategy"])
    agg_df.sort_values(by=sort_cols, inplace=True)
    agg_df.reset_index(drop=True, inplace=True)

    agg_df['Mean_Speedup'] = np.nan
    fp64_baselines = {}
    for group_key, eq_df in agg_df.groupby(entity_keys, observed=True):
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        mask = pd.Series(True, index=agg_df.index)
        for col, value in zip(entity_keys, key):
            mask &= agg_df[col] == value
        fp64_mask = eq_df['Strategy'].str.contains('fp64', case=False, na=False) & ~eq_df['Strategy'].str.contains('dynamic', case=False, na=False)
        if any(fp64_mask):
            fp64_run = eq_df[fp64_mask].iloc[0]
            fp64_baselines[key] = {'rRMSE': fp64_run['Mean_rRMSE'], 'rMAE': fp64_run['Mean_rMAE'], 'Time': fp64_run['Mean_Time']}
            agg_df.loc[mask, 'Mean_Speedup'] = fp64_run['Mean_Time'] / agg_df.loc[mask, 'Mean_Time']

    agg_df['is_green'] = False
    for idx, row in agg_df.iterrows():
        key = tuple(row[col] for col in entity_keys)
        if 'fp64' not in row['Strategy'].lower() and key in fp64_baselines:
            baseline = fp64_baselines[key]
            is_rrmse_green = pd.notnull(row['Mean_rRMSE']) and row['Mean_rRMSE'] <= 10 * baseline['rRMSE']
            is_rmae_green = pd.notnull(row['Mean_rMAE']) and row['Mean_rMAE'] <= 10 * baseline['rMAE']
            if is_rrmse_green and is_rmae_green:
                agg_df.loc[idx, 'is_green'] = True

    green_df = agg_df[agg_df['is_green']].copy()
    min_time_indices = green_df.groupby(entity_keys, observed=True)["Mean_Time"].idxmin()
    min_rrmse_indices = green_df.groupby(entity_keys, observed=True)["Mean_rRMSE"].idxmin()
    min_rmae_indices = green_df.groupby(entity_keys, observed=True)["Mean_rMAE"].idxmin()
    max_speedup_indices = green_df.groupby(entity_keys, observed=True)["Mean_Speedup"].idxmax()

    md_df = agg_df.copy().astype(object)
    latex_df = agg_df.copy().astype(object)

    def format_float(val, precision=4):
        if pd.isnull(val): return "-"
        return f"{val:.{precision}f}"

    for idx in agg_df.index:
        row_data = agg_df.loc[idx]
        eq_name = row_data['Equation']
        model_name = row_data['Model']
        strategy_name = row_data['Strategy']
        is_dynamic = "dynamic precision" in strategy_name.lower()

        str_time = f"{row_data['Mean_Time']:,.2f}" if pd.notnull(row_data['Mean_Time']) else "-"
        str_std_time = f"{row_data['Std_Time']:,.2f}" if pd.notnull(row_data['Std_Time']) else ""
        rrmse_str = format_float(row_data['Mean_rRMSE'], 4)
        std_rrmse_str = format_float(row_data['Std_rRMSE'], 4) if pd.notnull(row_data['Std_rRMSE']) else ""
        rmae_str = format_float(row_data['Mean_rMAE'], 4)
        std_rmae_str = format_float(row_data['Std_rMAE'], 4) if pd.notnull(row_data['Std_rMAE']) else ""
        speedup_str = f"{row_data['Mean_Speedup']:.2f}" if pd.notnull(row_data['Mean_Speedup']) else "-"

        rrmse_color_prefix = ""
        rmae_color_prefix = ""
        key = tuple(row_data[col] for col in entity_keys)
        if 'fp64' not in strategy_name.lower() and key in fp64_baselines:
            baseline = fp64_baselines[key]
            if pd.notnull(row_data['Mean_rRMSE']) and pd.notnull(baseline['rRMSE']):
                rrmse_color_prefix = "\\cellcolor{green!20}" if row_data['Mean_rRMSE'] <= 10 * baseline['rRMSE'] else "\\cellcolor{orange!50}"
            if pd.notnull(row_data['Mean_rMAE']) and pd.notnull(baseline['rMAE']):
                rmae_color_prefix = "\\cellcolor{green!20}" if row_data['Mean_rMAE'] <= 10 * baseline['rMAE'] else "\\cellcolor{orange!50}"

        def assemble_cell(val_str, std_str, is_best, color_prefix="", is_speedup=False):
            if val_str == "-": return "-", "-"
            base_md = f"{val_str}x" if is_speedup else f"{val_str} ± {std_str}" if std_str else val_str
            md_str = f"**{base_md}**" if is_best else base_md
            if is_dynamic: md_str = f"<span style='color:blue'>{md_str}</span>"

            v_tex = f"\\mathbf{{{val_str}}}" if is_best else val_str
            if std_str:
                s_tex = f"\\mathbf{{{std_str}}}" if is_best else std_str
                tex_inner = f"{v_tex} \\pm {s_tex}"
            else:
                tex_inner = f"{v_tex}\\times" if is_speedup else f"{v_tex}"

            tex_str = f"\\({tex_inner}\\)"
            if is_dynamic: tex_str = f"\\textcolor{{blue}}{{{tex_str}}}"
            if color_prefix: tex_str = f"{color_prefix}{tex_str}"
            return md_str, tex_str

        md_df.loc[idx, 'Mean_Time'], latex_df.loc[idx, 'Mean_Time'] = assemble_cell(str_time, str_std_time, (idx in min_time_indices.values))
        md_df.loc[idx, 'Mean_rRMSE'], latex_df.loc[idx, 'Mean_rRMSE'] = assemble_cell(rrmse_str, std_rrmse_str, (idx in min_rrmse_indices.values), rrmse_color_prefix)
        md_df.loc[idx, 'Mean_rMAE'], latex_df.loc[idx, 'Mean_rMAE'] = assemble_cell(rmae_str, std_rmae_str, (idx in min_rmae_indices.values), rmae_color_prefix)
        md_df.loc[idx, 'Mean_Speedup'], latex_df.loc[idx, 'Mean_Speedup'] = assemble_cell(speedup_str, "", (idx in max_speedup_indices.values), is_speedup=True)

        if is_dynamic:
            md_df.loc[idx, 'Equation'] = f"<span style='color:blue'>{eq_name}</span>"
            latex_df.loc[idx, 'Equation'] = f"\\textcolor{{blue}}{{{eq_name}}}"
            md_df.loc[idx, 'Strategy'] = f"<span style='color:blue'>{strategy_name}</span>"
            latex_df.loc[idx, 'Strategy'] = f"\\textcolor{{blue}}{{{strategy_name}}}"

    num_seeds = int(agg_df['Run_Count'].max()) if not agg_df.empty else 0
    md_caption_text = f"Average result from {num_seeds} random seeds. Results in the same order of magnitude compared with FP64 are highlighted in green."
    tex_caption_text = f"Average result from {num_seeds} random seeds. Results in the same order of magnitude compared with FP64 are highlighted in green."

    cols_to_drop = ['Model', 'Std_Time', 'Std_rRMSE', 'Std_rMAE', 'Run_Count', 'is_green']
    md_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    latex_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    rename_cols = {"Mean_Time": "Avg Time (s)", "Mean_rRMSE": "rRMSE", "Mean_rMAE": "rMAE", "Mean_Speedup": "Speedup vs FP64"}
    md_df.rename(columns=rename_cols, inplace=True)
    latex_df.rename(columns=rename_cols, inplace=True)

    display_cols = ["Equation", "Strategy", "Avg Time (s)", "rRMSE", "rMAE", "Speedup vs FP64"]
    if group_col is not None:
        display_cols = [group_col] + display_cols
    md_df = md_df[display_cols]
    latex_df = latex_df[display_cols]

    print("\n\n" + "=" * 60)
    print("--- Experiment Results Summary (Markdown) ---")
    print("=" * 60)
    print(f"**Caption:** *{md_caption_text}*\n")
    print(md_df.to_markdown(index=False))

    print("\n\n" + "=" * 60)
    print("--- Experiment Results Summary (LaTeX) ---")
    print("=" * 60)
    print("% Note: Make sure to include \\usepackage{xcolor}, \\usepackage{amsmath} and \\usepackage{colortbl} in your LaTeX preamble!")
    latex_str = latex_df.to_latex(index=False, escape=False, caption=tex_caption_text)

    lines = latex_str.split('\n')
    processed_lines = []
    row_idx = 0
    in_body = False
    separator_cols = ["Equation"] if group_col is None else [group_col, "Equation"]
    for line in lines:
        if '\\midrule' in line: in_body = True
        if '\\bottomrule' in line: in_body = False
        if (
            in_body and '&' in line and row_idx > 0
            and any(agg_df[col].iloc[row_idx] != agg_df[col].iloc[row_idx - 1] for col in separator_cols)
        ):
            processed_lines.append('\\hline')
        processed_lines.append(line)
        if in_body and '&' in line: row_idx += 1
    print('\n'.join(processed_lines))

    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    def get_seed_speedup(row):
        key = tuple(row[col] for col in entity_keys)
        if key in baseline_strats:
            b_strat = baseline_strats[key]
            match_mask = (df['Seed'] == row['Seed']) & (df['Strategy'] == b_strat)
            for col in entity_keys:
                match_mask &= df[col] == row[col]
            match = df[match_mask]
            if not match.empty:
                baseline_time = match.iloc[0]['Training Time (s)']
                if row['Training Time (s)'] > 0:
                    return baseline_time / row['Training Time (s)']
        return np.nan

    df['Speedup_vs_FP64'] = df.apply(get_seed_speedup, axis=1)

    plot_groups = df.groupby('Model') if group_col is None else df.groupby([group_col, "Model"])
    for plot_key, model_df in plot_groups:
        if group_col is None:
            model_name = plot_key
            plot_model_name = model_name
            print(f"\n--- Generating plots for model: {model_name} ---")
        else:
            group_value, model_name = plot_key
            plot_model_name = f"{model_name}_{safe_tag(group_col)}_{safe_tag(group_value)}"
            print(f"\n--- Generating plots for model: {model_name} | {group_col}: {group_value} ---")

        model_run_ids = set(model_df['wandb_run'].apply(lambda r: r.id))
        model_all_runs_data = [r for r in all_runs_data if r['wandb_run'].id in model_run_ids]

        plot_training_time_vs_accuracy(model_df, plot_model_name)
        plot_l2_vs_walltime(model_df, model_all_runs_data, plot_model_name)
        plot_l2_vs_step(model_df, model_all_runs_data, plot_model_name)
        plot_log_proxy_vs_step(model_df, model_all_runs_data, plot_model_name)


    print("\nPlots generated successfully!")

def analyze_hyperparameter_groups():
    """
    Fetches all runs, groups them by project, equation, and strategy,
    and generates a LaTeX summary table for each project analyzing performance and convergence.
    """
    import wandb
    import pandas as pd
    import numpy as np
    from matplotlib.lines import Line2D

    api = wandb.Api()

    ENTITY = "lokious-wageningen-uinversity"
    PROJECTS = [
        "PINN_heat_dynamic_precision_MLP"
        # "pinn_wave_dynamic_precision_PINNMamba",
        # "pinn_reaction_dynamic_precision_PINNMamba",
        # "pinn_convection_dynamic_precision_PINNMamba",
        # "pinn_allen_cahn_dynamic_precision_PINNMamba",
        # "pinn_convection_dynamic_precision_PINNsFormer",
        # "pinn_reaction_dynamic_precision_PINNsFormer",
        # "pinn_wave_dynamic_precision_PINNsFormer",
        # "pinn_allen_cahn_dynamic_precision_PINNsFormer",
        # "pinn_wave_dynamic_precision_KAN",
        # "pinn_reaction_dynamic_precision_KAN",
        # "pinn_convection_dynamic_precision_KAN",
        # "pinn_allen_cahn_dynamic_precision_KAN",
    ]
    CONVERGENCE_THRESHOLD = 0.3 #1.5 for all results becasue some equations are not converge fo all seeds.

    print(f"Fetching runs from entity: '{ENTITY}'...")

    all_runs_data = []

    for project in PROJECTS:
        runs = api.runs(path=f"{ENTITY}/{project}")
        print(f"Processing {len(runs)} runs in project '{project}'...")

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            if "rMAE" not in run.summary or "rRMSE" not in run.summary or "training_time_seconds" not in run.summary:
                continue

            config = run.config
            summary = run.summary

            raw_strategy = config.get("strategy", "N/A")
            if "switching" in raw_strategy.lower():
                continue

            if raw_strategy == "fp64_curvature":
                raw_strategy = "fp64"
            elif raw_strategy == "fp32_curvature":
                raw_strategy = "fp32"

            strategy = raw_strategy.replace('_', ' ')
            if config.get("rescale_derivative", False):
                strategy += " + Rescale"

            if "dynamic" in strategy.lower() or "curvature" in strategy.lower():
                proxy_val = config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None))
                if proxy_val is None:
                    proxy_val = 2.0
                strategy += f" (Proxy {proxy_val})"

            run_data = {
                'Project': project,
                'Equation': project.split('_')[1].replace('-', ' ').title(),
                'Strategy': strategy,
                'rMAE': summary['rMAE'],
                'rRMSE': summary['rRMSE'],
                'Training Time (s)': summary['training_time_seconds']
            }
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("No valid runs found to analyze.")
        return

    df = pd.DataFrame(all_runs_data)
    df['is_converged'] = df['rMAE'] <= CONVERGENCE_THRESHOLD

    print("\n" + "=" * 80)
    print("Hyperparameter Group Analysis (LaTeX Output)")
    print(f"(A run is considered non-converged if rMAE > {CONVERGENCE_THRESHOLD})")
    print("=" * 80 + "\n")

    for project_name, project_df in df.groupby('Project'):
        if project_df.empty:
            continue

        convergence_stats = project_df.groupby(['Equation', 'Strategy']).agg(
            Total_Runs=('rMAE', 'size'),
            Converged_Runs=('is_converged', 'sum')
        ).reset_index()
        convergence_stats['Convergence Rate'] = (
                    convergence_stats['Converged_Runs'] / convergence_stats['Total_Runs']).apply(lambda x: f"{x:.0%}")

        converged_df = project_df[project_df['is_converged']]

        if not converged_df.empty:
            perf_stats = converged_df.groupby(['Equation', 'Strategy']).agg(
                Mean_Time=('Training Time (s)', 'mean'),
                Std_Time=('Training Time (s)', 'std'),
                Mean_rRMSE=('rRMSE', 'mean'),
                Std_rRMSE=('rRMSE', 'std'),
                Mean_rMAE=('rMAE', 'mean'),
                Std_rMAE=('rMAE', 'std'),
            ).reset_index()
        else:
            perf_stats = pd.DataFrame(columns=[
                'Equation', 'Strategy', 'Mean_Time', 'Std_Time',
                'Mean_rRMSE', 'Std_rRMSE', 'Mean_rMAE', 'Std_rMAE'
            ])

        summary_df = pd.merge(convergence_stats, perf_stats, on=['Equation', 'Strategy'], how='left')

        summary_df['Time (s)'] = summary_df.apply(
            lambda row: f"{row['Mean_Time']:.1f} $\\pm$ {row['Std_Time']:.1f}" if pd.notna(row['Mean_Time']) else "-",
            axis=1
        )
        summary_df['rRMSE'] = summary_df.apply(
            lambda row: f"{row['Mean_rRMSE']:.4f} $\\pm$ {row['Std_rRMSE']:.4f}" if pd.notna(
                row['Mean_rRMSE']) else "-", axis=1
        )
        summary_df['rMAE'] = summary_df.apply(
            lambda row: f"{row['Mean_rMAE']:.4f} $\\pm$ {row['Std_rMAE']:.4f}" if pd.notna(row['Mean_rMAE']) else "-",
            axis=1
        )

        final_cols_map = {
            'Equation': 'Equation',
            'Strategy': 'Strategy',
            'Time (s)': 'Avg Time (s)',
            'rRMSE': 'rRMSE',
            'rMAE': 'rMAE',
            'Convergence Rate': 'Convergence'
        }
        final_df = summary_df[list(final_cols_map.keys())].rename(columns=final_cols_map)

        clean_project_name = project_name.replace('_', ' ').title()
        table_label = f"tab:hyperparams_{project_name.replace('_', '')}"

        latex_table = final_df.to_latex(
            index=False,
            escape=False,
            caption=f"Performance and convergence analysis for {clean_project_name}.",
            label=table_label,
            column_format='l' * 2 + 'r' * (len(final_df.columns) - 2)
        )

        print(f"% --- LaTeX Table for {project_name} ---")
        print(latex_table)
        print("\n" + "-" * 80 + "\n")


def analyze_hyperparameter_groups_PDE_system():
    """
    Fetches all runs, groups them by project, equation, and strategy,
    and generates a LaTeX summary table for each project analyzing performance and convergence.
    """
    import wandb
    import pandas as pd
    import numpy as np
    from matplotlib.lines import Line2D

    api = wandb.Api()

    ENTITY = "lokious-wageningen-uinversity"
    PROJECTS = [
        "pinn_beam_dynamic_precision_plotMLP",
        "pinn_ns2dc_dynamic_precision_PINN",

    ]
    CONVERGENCE_THRESHOLD = 0.3  # 1.5 for all results because some equations are not converge for all seeds.

    def safe_tag(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

    def strategy_color(strategy):
        strategy_lower = str(strategy).lower()
        if "fp64" in strategy_lower and "dynamic" not in strategy_lower:
            return "#1F77B4"
        if "fp32" in strategy_lower and "dynamic" not in strategy_lower:
            return "#7F7F7F"
        if "dynamic" in strategy_lower or "curvature" in strategy_lower:
            return "#FF7F0E"
        return "#2CA02C"

    def strategy_marker(strategy):
        strategy_lower = str(strategy).lower()
        if "dynamic" in strategy_lower or "curvature" in strategy_lower:
            return "s"
        if "fp32" in strategy_lower:
            return "^"
        return "o"

    def is_fp64_baseline(strategy):
        strategy_lower = str(strategy).lower()
        return "fp64" in strategy_lower and "dynamic" not in strategy_lower

    def is_fp32_strategy(strategy):
        strategy_lower = str(strategy).lower()
        return "fp32" in strategy_lower and "dynamic" not in strategy_lower

    def strategy_label(strategy):
        strategy_lower = str(strategy).lower()
        if "dynamic" in strategy_lower or "curvature" in strategy_lower:
            return "dynamic precision"
        if "fp64" in strategy_lower:
            return "fp64"
        if "fp32" in strategy_lower:
            return "fp32"
        return str(strategy)

    def fetch_log_proxy_history(run):
        keys = ["_step", "smoothed_log_proxy"]
        try:
            history = pd.DataFrame(list(run.scan_history(keys=keys, page_size=10000)))
        except Exception:
            try:
                history = run.history(keys=keys, samples=100000)
            except Exception:
                return pd.DataFrame()

        if history.empty or "_step" not in history.columns or "smoothed_log_proxy" not in history.columns:
            return pd.DataFrame()

        history = history[["_step", "smoothed_log_proxy"]].dropna().copy()
        history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
        history["smoothed_log_proxy"] = pd.to_numeric(history["smoothed_log_proxy"], errors="coerce")
        history = history.dropna()
        history = history[history["smoothed_log_proxy"] != 0.0]
        return history.sort_values("_step").reset_index(drop=True)

    def plot_pde_system_log_proxy_vs_step(project_runs, project_name):
        if not project_runs:
            return

        equations = sorted({run_data["Equation"] for run_data in project_runs})
        ncols = min(2, max(1, len(equations)))
        nrows = int(np.ceil(len(equations) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows), squeeze=False, sharey=True)
        axes_flat = axes.flatten()
        legend_handles = {}
        plotted_any = False

        for i, equation in enumerate(equations):
            ax = axes_flat[i]
            eq_runs = [run_data for run_data in project_runs if run_data["Equation"] == equation]
            strategies = sorted({run_data["Strategy"] for run_data in eq_runs})

            for strategy in strategies:
                strat_runs = [run_data for run_data in eq_runs if run_data["Strategy"] == strategy]
                histories = []
                for run_data in strat_runs:
                    history = fetch_log_proxy_history(run_data["wandb_run"])
                    if not history.empty:
                        histories.append(history)

                if not histories:
                    continue

                min_len = min(len(history) for history in histories)
                if min_len == 0:
                    continue

                aligned_steps = np.array([history["_step"].values[:min_len] for history in histories])
                aligned_proxy = np.array([history["smoothed_log_proxy"].values[:min_len] for history in histories])
                avg_steps = np.mean(aligned_steps, axis=0)
                mean_proxy = np.mean(aligned_proxy, axis=0)
                std_proxy = np.std(aligned_proxy, axis=0)
                color = strategy_color(strategy)
                label = strategy_label(strategy)

                line, = ax.plot(avg_steps, mean_proxy, color=color, label=label, linewidth=2.3)
                ax.fill_between(avg_steps, mean_proxy - std_proxy, mean_proxy + std_proxy, color=color, alpha=0.2)
                legend_handles.setdefault(label, line)
                plotted_any = True

            ax.set_title(equation, fontsize=22)
            ax.set_xlabel("Training Step", fontsize=20)
            ax.set_ylabel(r"$\tilde{z}_t$", fontsize=20)
            ax.set_ylim(1.5, 4.0)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", which="major", labelsize=14)

        for ax in axes_flat[len(equations):]:
            ax.axis("off")

        if not plotted_any:
            plt.close(fig)
            print(f"  Skipping log-proxy plot for {project_name}: no smoothed_log_proxy history.")
            return

        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="upper center",
                ncol=len(legend_handles),
                fontsize=21,
                title="Strategy",
                title_fontsize=20,
            )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        filename = f"pde_system_log_proxy_vs_step_{safe_tag(project_name)}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved log-proxy plot: {filename}")
        plt.close(fig)

        for equation in equations:
            eq_runs = [run_data for run_data in project_runs if run_data["Equation"] == equation]
            strategies = sorted({run_data["Strategy"] for run_data in eq_runs})
            dynamic_strategies = [
                strategy for strategy in strategies
                if "dynamic" in strategy.lower() or "curvature" in strategy.lower()
            ]
            baseline_strategies = [
                strategy for strategy in strategies
                if is_fp32_strategy(strategy) or is_fp64_baseline(strategy)
            ]

            for dynamic_strategy in dynamic_strategies:
                fig, ax = plt.subplots(figsize=(10, 6))
                plotted_individual = False
                for strategy in [*baseline_strategies, dynamic_strategy]:
                    strat_runs = [run_data for run_data in eq_runs if run_data["Strategy"] == strategy]
                    histories = []
                    for run_data in strat_runs:
                        history = fetch_log_proxy_history(run_data["wandb_run"])
                        if not history.empty:
                            histories.append(history)

                    if not histories:
                        continue

                    min_len = min(len(history) for history in histories)
                    if min_len == 0:
                        continue

                    aligned_steps = np.array([history["_step"].values[:min_len] for history in histories])
                    aligned_proxy = np.array([history["smoothed_log_proxy"].values[:min_len] for history in histories])
                    avg_steps = np.mean(aligned_steps, axis=0)
                    mean_proxy = np.mean(aligned_proxy, axis=0)
                    std_proxy = np.std(aligned_proxy, axis=0)
                    color = strategy_color(strategy)

                    ax.plot(avg_steps, mean_proxy, color=color, label=strategy_label(strategy), linewidth=2.2)
                    ax.fill_between(avg_steps, mean_proxy - std_proxy, mean_proxy + std_proxy, color=color, alpha=0.2)
                    plotted_individual = True

                if not plotted_individual:
                    plt.close(fig)
                    continue

                ax.set_xlabel("Training Step", fontsize=20)
                ax.set_ylabel(r"$\tilde{z}_t$", fontsize=20)
                ax.set_ylim(1.5, 4.0)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.legend(title="Strategy", fontsize=21)
                fig.tight_layout()

                proxy_match = re.search(r"\(Proxy (.*?)\)", dynamic_strategy)
                proxy_text = f"_proxy_{safe_tag(proxy_match.group(1))}" if proxy_match else "_dynamic"
                filename = f"pde_system_log_proxy_vs_step_{safe_tag(project_name)}_{safe_tag(equation)}{proxy_text}.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"  Saved log-proxy plot: {filename}")
                plt.close(fig)

    def plot_pde_system_speed_accuracy_scatter(project_df, project_name):
        plot_source = project_df[project_df["is_converged"]].copy()
        plot_source = plot_source[~plot_source["Strategy"].apply(is_fp32_strategy)].copy()
        if plot_source.empty:
            print(f"  Skipping speed-accuracy scatter for {project_name}: no non-FP32 converged runs.")
            return

        plot_source["Normalized Time (%)"] = np.nan
        for equation, eq_df in plot_source.groupby("Equation"):
            fp64_df = eq_df[eq_df["Strategy"].apply(is_fp64_baseline)]
            if fp64_df.empty:
                continue

            mean_fp64_time = fp64_df["Training Time (s)"].mean()
            if pd.isna(mean_fp64_time) or mean_fp64_time <= 0:
                continue

            mask = plot_source["Equation"] == equation
            plot_source.loc[mask, "Normalized Time (%)"] = (
                plot_source.loc[mask, "Training Time (s)"] / mean_fp64_time
            ) * 100

        plot_df = plot_source.dropna(subset=["Normalized Time (%)", "System_rMAE"]).copy()
        if plot_df.empty:
            print(f"  Skipping speed-accuracy scatter for {project_name}: no FP64-normalized plottable data.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        for strategy in sorted(plot_df["Strategy"].unique()):
            strat_df = plot_df[plot_df["Strategy"] == strategy]
            color = strategy_color(strategy)
            marker = strategy_marker(strategy)

            ax.scatter(
                strat_df["Normalized Time (%)"],
                strat_df["System_rMAE"],
                c=color,
                marker=marker,
                s=150,
                alpha=0.65,
                edgecolors="black",
                linewidth=1.5,
            )

            mean_x = strat_df["Normalized Time (%)"].mean()
            mean_y = strat_df["System_rMAE"].mean()
            ax.scatter(
                mean_x,
                mean_y,
                c=color,
                marker=marker,
                s=420,
                alpha=1.0,
                edgecolors="black",
                linewidth=2.5,
                zorder=5,
            )
            ax.annotate(strategy_label(strategy), (mean_x, mean_y), textcoords="offset points", xytext=(8, 6), fontsize=10)

        time_vals = plot_df["Normalized Time (%)"].values
        error_vals = plot_df["System_rMAE"].values
        pareto_mask = np.ones(len(plot_df), dtype=bool)
        for i in range(len(plot_df)):
            for j in range(len(plot_df)):
                if i != j and time_vals[j] < time_vals[i] and error_vals[j] < error_vals[i]:
                    pareto_mask[i] = False
                    break

        pareto_df = plot_df[pareto_mask]
        if len(pareto_df) > 1:
            pareto_sorted = pareto_df.sort_values("Normalized Time (%)")
            ax.plot(
                pareto_sorted["Normalized Time (%)"],
                pareto_sorted["System_rMAE"],
                linestyle="--",
                color="black",
                alpha=0.3,
                linewidth=2.0,
                zorder=1,
            )

        ax.axvline(x=100, color="red", linestyle=":", alpha=0.6, linewidth=2.5, zorder=0)
        ax.set_xlim(0, max(150, plot_df["Normalized Time (%)"].max() * 1.1))
        ax.set_xlabel("Relative Training Time (% of FP64 Mean)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Mean state rMAE (Error)", fontsize=16, fontweight="bold")
        ax.set_title(f"PDE-System Speed vs Accuracy | {project_name}", fontsize=17, pad=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4",
                   markersize=12, markeredgecolor="black", label="fp64"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF7F0E",
                   markersize=12, markeredgecolor="black", label="dynamic precision"),
            Line2D([0], [0], color="red", linestyle=":", linewidth=2.5, label="FP64 Benchmark (100%)"),
        ]
        ax.legend(handles=legend_handles, fontsize=11, loc="upper left", framealpha=0.95)
        fig.tight_layout()

        filename = f"pde_system_speed_accuracy_scatter_{safe_tag(project_name)}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved speed-accuracy scatter: {filename}")
        plt.close(fig)

    print(f"Fetching runs from entity: '{ENTITY}'...")

    all_runs_data = []

    for project in PROJECTS:
        runs = api.runs(path=f"{ENTITY}/{project}")
        print(f"Processing {len(runs)} runs in project '{project}'...")

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            # Check for the exact 4 metrics + time
            required_metrics = ["w_rMAE", "w_rRMSE", "theta_rMAE", "theta_rRMSE", "training_time_seconds"]
            if not all(metric in run.summary for metric in required_metrics):
                continue

            config = run.config
            summary = run.summary

            raw_strategy = str(config.get("strategy", "N/A"))
            if "switching" in raw_strategy.lower():
                continue

            if raw_strategy == "fp64_curvature":
                raw_strategy = "fp64"
            elif raw_strategy == "fp32_curvature":
                raw_strategy = "fp32"

            strategy = raw_strategy.replace('_', ' ')
            if config.get("rescale_derivative", False):
                strategy += " + Rescale"

            if "dynamic" in strategy.lower() or "curvature" in strategy.lower():
                proxy_val = config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None))
                if proxy_val is None:
                    proxy_val = 2.0
                strategy += f" (Proxy {proxy_val})"

            # Store the correct metrics
            system_rrmse = float(np.mean([float(summary['w_rRMSE']), float(summary['theta_rRMSE'])]))
            system_rmae = float(np.mean([float(summary['w_rMAE']), float(summary['theta_rMAE'])]))
            run_data = {
                'Project': project,
                'Equation': project.split('_')[1].replace('-', ' ').title(),
                'Raw_Strategy': raw_strategy,
                'Strategy': strategy,
                'LOG_PROXY_LOW': config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None)),
                'Seed': config.get("seed", -1),
                'w_rRMSE': summary['w_rRMSE'],
                'w_rMAE': summary['w_rMAE'],
                'theta_rRMSE': summary['theta_rRMSE'],
                'theta_rMAE': summary['theta_rMAE'],
                'System_rRMSE': system_rrmse,
                'System_rMAE': system_rmae,
                'rRMSE': system_rrmse,
                'rMAE': system_rmae,
                'Training Time (s)': summary['training_time_seconds'],
                'wandb_run': run,
            }
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("No valid runs found to analyze.")
        return

    df = pd.DataFrame(all_runs_data)
    numeric_cols = [
        'Training Time (s)', 'w_rRMSE', 'w_rMAE', 'theta_rRMSE', 'theta_rMAE',
        'System_rRMSE', 'System_rMAE', 'rRMSE', 'rMAE',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Training Time (s)', 'w_rRMSE', 'w_rMAE', 'theta_rRMSE', 'theta_rMAE'], inplace=True)

    # Require BOTH rMAEs to be under the threshold for convergence
    df['is_converged'] = (df['w_rMAE'] <= CONVERGENCE_THRESHOLD) & (df['theta_rMAE'] <= CONVERGENCE_THRESHOLD)

    print("\n" + "=" * 80)
    print("Hyperparameter Group Analysis (LaTeX Output)")
    print(f"(A run is considered non-converged if either w_rMAE or theta_rMAE > {CONVERGENCE_THRESHOLD})")
    print("=" * 80 + "\n")

    for project_name, project_df in df.groupby('Project'):
        if project_df.empty:
            continue

        convergence_stats = project_df.groupby(['Equation', 'Strategy']).agg(
            Total_Runs=('w_rMAE', 'size'),
            Converged_Runs=('is_converged', 'sum')
        ).reset_index()
        convergence_stats['Convergence Rate'] = (
                convergence_stats['Converged_Runs'] / convergence_stats['Total_Runs']).apply(lambda x: f"{x:.0%}")

        converged_df = project_df[project_df['is_converged']]

        # Aggregate the 4 updated metrics
        if not converged_df.empty:
            perf_stats = converged_df.groupby(['Equation', 'Strategy']).agg(
                Mean_Time=('Training Time (s)', 'mean'),
                Std_Time=('Training Time (s)', 'std'),
                Mean_w_rRMSE=('w_rRMSE', 'mean'),
                Std_w_rRMSE=('w_rRMSE', 'std'),
                Mean_w_rMAE=('w_rMAE', 'mean'),
                Std_w_rMAE=('w_rMAE', 'std'),
                Mean_theta_rRMSE=('theta_rRMSE', 'mean'),
                Std_theta_rRMSE=('theta_rRMSE', 'std'),
                Mean_theta_rMAE=('theta_rMAE', 'mean'),
                Std_theta_rMAE=('theta_rMAE', 'std'),
            ).reset_index()
        else:
            perf_stats = pd.DataFrame(columns=[
                'Equation', 'Strategy', 'Mean_Time', 'Std_Time',
                'Mean_w_rRMSE', 'Std_w_rRMSE', 'Mean_w_rMAE', 'Std_w_rMAE',
                'Mean_theta_rRMSE', 'Std_theta_rRMSE', 'Mean_theta_rMAE', 'Std_theta_rMAE'
            ])

        summary_df = pd.merge(convergence_stats, perf_stats, on=['Equation', 'Strategy'], how='left')

        # Format Time
        summary_df['Time (s)'] = summary_df.apply(
            lambda row: f"{row['Mean_Time']:.1f} $\\pm$ {row['Std_Time']:.1f}" if pd.notna(row['Mean_Time']) else "-",
            axis=1
        )

        # Format the 4 metrics with mean ± std
        for metric in ['w_rRMSE', 'w_rMAE', 'theta_rRMSE', 'theta_rMAE']:
            mean_col = f'Mean_{metric}'
            std_col = f'Std_{metric}'

            summary_df[metric] = summary_df.apply(
                lambda row, m=mean_col, s=std_col: f"{row[m]:.4f} $\\pm$ {row[s]:.4f}" if pd.notna(row[m]) else "-",
                axis=1
            )

        # Map to final LaTeX column names
        final_cols_map = {
            'Equation': 'Equation',
            'Strategy': 'Strategy',
            'Time (s)': 'Avg Time (s)',
            'w_rRMSE': r'w\_rRMSE',
            'w_rMAE': r'w\_rMAE',
            'theta_rRMSE': r'$\theta$\_rRMSE',
            'theta_rMAE': r'$\theta$\_rMAE',
            'Convergence Rate': 'Convergence'
        }

        final_df = summary_df[list(final_cols_map.keys())].rename(columns=final_cols_map)

        clean_project_name = project_name.replace('_', ' ').title()
        table_label = f"tab:hyperparams_{project_name.replace('_', '')}"

        latex_table = final_df.to_latex(
            index=False,
            escape=False,
            caption=f"Performance and convergence analysis for {clean_project_name}.",
            label=table_label,
            column_format='l' * 2 + 'r' * (len(final_df.columns) - 2)
        )

        print(f"% --- LaTeX Table for {project_name} ---")
        print(latex_table)
        print("\n" + "-" * 80 + "\n")

        project_runs_data = [
            run_data for run_data in all_runs_data
            if run_data['Project'] == project_name
        ]
        plot_pde_system_log_proxy_vs_step(project_runs_data, project_name)
        plot_pde_system_speed_accuracy_scatter(project_df, project_name)


def analyze_hyperparameter_groups_ns2d_c():
    """
    Fetches NS2d-C runs, groups them by project, lid amplitude, and strategy,
    and generates a LaTeX summary table plus per-amplitude plots.
    """
    import wandb
    import pandas as pd
    import numpy as np
    from matplotlib.lines import Line2D

    api = wandb.Api()

    ENTITY = "lokious-wageningen-uinversity"
    PROJECTS = [
        "pinn_ns2dc_dynamic_precision_PINN",
    ]
    CONVERGENCE_THRESHOLD = 0.3
    COMPONENT_METRICS = [
        "u_rRMSE", "v_rRMSE", "p_rRMSE",
        "u_rMAE", "v_rMAE", "p_rMAE",
    ]
    OVERALL_METRICS = ["rRMSE", "rMAE"]
    TABLE_METRICS = [*OVERALL_METRICS, *COMPONENT_METRICS]

    def clean_strategy(raw_strat):
        if pd.isna(raw_strat):
            return "N/A"
        raw_strat = str(raw_strat)
        if raw_strat == "fp64_curvature":
            return "fp64"
        if raw_strat == "fp32_curvature":
            return "fp32"
        if raw_strat in ["dynamic", "dynamic_precision_lbfgs", "dynamic precision lbfgs"]:
            return "dynamic"
        return raw_strat.replace("_", " ")

    def format_amplitude_tag(value):
        if pd.isna(value):
            return "unknown"
        value = float(value)
        if value.is_integer():
            return f"a{int(value)}"
        return f"a{str(value).replace('.', '_')}"

    def format_amplitude_title(value):
        if pd.isna(value):
            return "unknown"
        value = float(value)
        return f"a = {value:g}"

    def strategy_color(strategy):
        strategy_lower = str(strategy).lower()
        if "fp64" in strategy_lower:
            return "#1F77B4"
        if "fp32" in strategy_lower:
            return "#7F7F7F"
        if "dynamic" in strategy_lower or "curvature" in strategy_lower:
            return "#FF7F0E"
        return "#2CA02C"

    def is_fp64_baseline(strategy):
        strategy_lower = str(strategy).lower()
        return "fp64" in strategy_lower and "dynamic" not in strategy_lower

    def strategy_marker(strategy):
        strategy_lower = str(strategy).lower()
        if "dynamic" in strategy_lower or "curvature" in strategy_lower:
            return "s"
        if "fp32" in strategy_lower:
            return "^"
        return "o"

    def is_fp32_strategy(strategy):
        return "fp32" in str(strategy).lower()

    def summary_metric(summary, metric):
        value = summary.get(metric)
        if value is None:
            value = summary.get(f"final_{metric}", np.nan)
        return value

    def format_mean_std(row, metric, precision=4):
        mean = row.get(f"Mean_{metric}")
        if pd.isna(mean):
            return "-"

        std = row.get(f"Std_{metric}")
        mean_text = f"{mean:.{precision}f}"
        if pd.isna(std):
            return mean_text
        return f"{mean_text} $\\pm$ {std:.{precision}f}"

    def format_speedup(value):
        if pd.isna(value):
            return "-"
        return f"{value:.2f}x"

    def latex_metric_label(metric):
        return metric.replace("_", r"\_")

    def plot_amplitude_summary(amplitude_df, project_name, amplitude_value):
        if "is_converged" in amplitude_df.columns:
            plot_source = amplitude_df[amplitude_df["is_converged"]].copy()
        else:
            plot_source = amplitude_df.copy()

        plot_source = plot_source[~plot_source["Strategy"].apply(is_fp32_strategy)].copy()

        if plot_source.empty:
            print(f"  Skipping scatter plot for {format_amplitude_title(amplitude_value)}: no non-FP32 converged runs.")
            return

        fp64_df = plot_source[plot_source["Strategy"].apply(is_fp64_baseline)]
        if fp64_df.empty:
            print(f"  Skipping scatter plot for {format_amplitude_title(amplitude_value)}: no FP64 baseline.")
            return

        plot_df = plot_source.dropna(subset=["Training Time (s)", "rMAE"]).copy()
        if plot_df.empty:
            print(f"  Skipping scatter plot for {format_amplitude_title(amplitude_value)}: no plottable rMAE data.")
            return

        mean_fp64_time = fp64_df["Training Time (s)"].mean()
        plot_df["Normalized Time (%)"] = (plot_df["Training Time (s)"] / mean_fp64_time) * 100

        fig, ax = plt.subplots(figsize=(12, 8))

        for strategy in sorted(plot_df["Strategy"].unique()):
            strat_df = plot_df[plot_df["Strategy"] == strategy]
            color = strategy_color(strategy)
            marker = strategy_marker(strategy)

            ax.scatter(
                strat_df["Normalized Time (%)"],
                strat_df["rMAE"],
                c=color,
                marker=marker,
                s=150,
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
            )

            mean_x = strat_df["Normalized Time (%)"].mean()
            mean_y = strat_df["rMAE"].mean()
            ax.scatter(
                mean_x,
                mean_y,
                c=color,
                marker=marker,
                s=420,
                alpha=1.0,
                edgecolors="black",
                linewidth=2.5,
                zorder=5,
            )
            ax.annotate(
                strategy,
                (mean_x, mean_y),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=10,
            )

        time_vals = plot_df["Normalized Time (%)"].values
        rmae_vals = plot_df["rMAE"].values
        pareto_mask = np.ones(len(plot_df), dtype=bool)
        for i in range(len(plot_df)):
            for j in range(len(plot_df)):
                if i != j and time_vals[j] < time_vals[i] and rmae_vals[j] < rmae_vals[i]:
                    pareto_mask[i] = False
                    break

        pareto_df = plot_df[pareto_mask]
        if len(pareto_df) > 1:
            pareto_sorted = pareto_df.sort_values("Normalized Time (%)")
            ax.plot(
                pareto_sorted["Normalized Time (%)"],
                pareto_sorted["rMAE"],
                linestyle="--",
                color="black",
                alpha=0.3,
                linewidth=2.0,
                zorder=1,
            )

        ax.axvline(x=100, color="red", linestyle=":", alpha=0.6, linewidth=2.5, zorder=0)
        ax.set_xlim(0, max(150, plot_df["Normalized Time (%)"].max() * 1.1))
        ax.set_xlabel("Relative Training Time (% of FP64 Mean)", fontsize=16, fontweight="bold")
        ax.set_ylabel("rMAE (Error)", fontsize=16, fontweight="bold")
        ax.set_title(
            f"NS2d-C Speed vs Accuracy | {format_amplitude_title(amplitude_value)}",
            fontsize=17,
            fontweight="bold",
            pad=14,
        )
        ax.grid(True, alpha=0.3, linestyle="--")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4", markersize=12, markeredgecolor="black", label="FP64"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF7F0E", markersize=12, markeredgecolor="black", label="Dynamic"),
            Line2D([0], [0], color="red", linestyle=":", linewidth=2.5, label="FP64 Benchmark (100%)"),
        ]
        ax.legend(handles=legend_handles, fontsize=11, loc="upper left", framealpha=0.95)
        fig.tight_layout()

        amplitude_tag = format_amplitude_tag(amplitude_value)
        filename = f"ns2dc_speed_accuracy_scatter_{project_name}_{amplitude_tag}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved amplitude-specific plot: {filename}")
        plt.close(fig)

    print(f"Fetching runs from entity: '{ENTITY}'...")

    all_runs_data = []

    for project in PROJECTS:
        runs = api.runs(path=f"{ENTITY}/{project}")
        print(f"Processing {len(runs)} runs in project '{project}'...")

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            required_metrics = ["rRMSE", "rMAE", "training_time_seconds"]
            if not all(metric in run.summary for metric in required_metrics):
                continue

            config = run.config
            summary = run.summary

            raw_strategy = config.get("strategy", "N/A")
            if "switching" in str(raw_strategy).lower():
                continue

            strategy = clean_strategy(raw_strategy)
            if config.get("rescale_derivative", False):
                strategy += " + Rescale"

            if "dynamic" in strategy.lower() or "curvature" in strategy.lower():
                proxy_val = config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None))
                if proxy_val is None:
                    proxy_val = 2.0
                strategy += f" (Proxy {proxy_val})"

            lid_amplitude_a = config.get("lid_amplitude_a", config.get("lid_amplitude", np.nan))
            if pd.isna(lid_amplitude_a):
                lid_amplitude_a = 4.0

            run_data = {
                "Project": project,
                "Model": project.split("_")[-1],
                "lid_amplitude_a": lid_amplitude_a,
                "Strategy": strategy,
                "rRMSE": summary["rRMSE"],
                "rMAE": summary["rMAE"],
                "Training Time (s)": summary["training_time_seconds"],
                "Seed": config.get("seed", -1),
            }
            for metric in COMPONENT_METRICS:
                run_data[metric] = summary_metric(summary, metric)
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("No valid NS2d-C runs found to analyze.")
        return

    df = pd.DataFrame(all_runs_data)
    df["Training Time (s)"] = pd.to_numeric(df["Training Time (s)"], errors="coerce")
    for metric in ["rRMSE", "rMAE", *COMPONENT_METRICS]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["lid_amplitude_a"] = pd.to_numeric(df["lid_amplitude_a"], errors="coerce").fillna(4.0)
    df.dropna(subset=["Training Time (s)", "rRMSE", "rMAE", "lid_amplitude_a"], inplace=True)

    df["is_converged"] = df["rMAE"] <= CONVERGENCE_THRESHOLD

    print("\n" + "=" * 80)
    print("NS2d-C Hyperparameter Group Analysis (LaTeX Output)")
    print(f"(A run is considered non-converged if rMAE > {CONVERGENCE_THRESHOLD})")
    print("=" * 80 + "\n")

    for project_name, project_df in df.groupby("Project"):
        if project_df.empty:
            continue

        for lid_value, amplitude_df in project_df.groupby("lid_amplitude_a"):
            if amplitude_df.empty:
                continue

            print(f"\n[{project_name}] lid_amplitude_a = {lid_value:g}")
            plot_amplitude_summary(amplitude_df, project_name, lid_value)

            convergence_stats = amplitude_df.groupby(["Strategy"]).agg(
                Total_Runs=("rMAE", "size"),
                Converged_Runs=("is_converged", "sum"),
            ).reset_index()
            convergence_stats["Convergence Rate"] = (
                convergence_stats["Converged_Runs"] / convergence_stats["Total_Runs"]
            ).apply(lambda x: f"{x:.0%}")

            converged_df = amplitude_df[amplitude_df["is_converged"]]
            metric_aggs = {
                f"Mean_{metric}": (metric, "mean")
                for metric in TABLE_METRICS
            }
            metric_aggs.update({
                f"Std_{metric}": (metric, "std")
                for metric in TABLE_METRICS
            })

            if not converged_df.empty:
                perf_stats = converged_df.groupby(["Strategy"]).agg(
                    Mean_Time=("Training Time (s)", "mean"),
                    Std_Time=("Training Time (s)", "std"),
                    **metric_aggs,
                ).reset_index()
            else:
                metric_stat_cols = [
                    f"{prefix}_{metric}"
                    for metric in TABLE_METRICS
                    for prefix in ("Mean", "Std")
                ]
                perf_stats = pd.DataFrame(columns=[
                    "Strategy", "Mean_Time", "Std_Time",
                    *metric_stat_cols,
                ])

            summary_df = pd.merge(convergence_stats, perf_stats, on=["Strategy"], how="left")

            fp64_summary = summary_df[
                summary_df["Strategy"].apply(is_fp64_baseline) & summary_df["Mean_Time"].notna()
            ]
            summary_df["Mean_Speedup"] = np.nan
            if not fp64_summary.empty:
                fp64_mean_time = fp64_summary.iloc[0]["Mean_Time"]
                summary_df["Mean_Speedup"] = fp64_mean_time / summary_df["Mean_Time"]

            summary_df["Time (s)"] = summary_df.apply(
                lambda row: format_mean_std(row, "Time", precision=1),
                axis=1
            )
            for metric in TABLE_METRICS:
                summary_df[metric] = summary_df.apply(
                    lambda row, metric=metric: format_mean_std(row, metric),
                    axis=1
                )
            summary_df["Mean_Speedup"] = summary_df["Mean_Speedup"].apply(format_speedup)

            def make_final_df(metrics):
                final_cols_map = {
                    "Strategy": "Strategy",
                    "Time (s)": "Avg Time (s)",
                    **{metric: latex_metric_label(metric) for metric in metrics},
                    "Mean_Speedup": r"Mean\_Speedup",
                    "Convergence Rate": "Convergence",
                    "Total_Runs": "Runs",
                }
                return summary_df[list(final_cols_map.keys())].rename(columns=final_cols_map)

            clean_project_name = project_name.replace("_", " ").title()
            amplitude_tag = format_amplitude_tag(lid_value)

            overall_df = make_final_df(OVERALL_METRICS)
            component_df = make_final_df(COMPONENT_METRICS)

            overall_latex_table = overall_df.to_latex(
                index=False,
                escape=False,
                caption=f"Overall performance and convergence analysis for {clean_project_name} at {format_amplitude_title(lid_value)}.",
                label=f"tab:ns2dc_{project_name.replace('_', '')}_{amplitude_tag}_overall",
                column_format='l' + 'r' * (len(overall_df.columns) - 1)
            )
            component_latex_table = component_df.to_latex(
                index=False,
                escape=False,
                caption=f"Component-wise performance and convergence analysis for {clean_project_name} at {format_amplitude_title(lid_value)}.",
                label=f"tab:ns2dc_{project_name.replace('_', '')}_{amplitude_tag}_components",
                column_format='l' + 'r' * (len(component_df.columns) - 1)
            )

            print(f"% --- Overall LaTeX Table for {project_name} ({format_amplitude_title(lid_value)}) ---")
            print(overall_latex_table)
            print("\n" + "-" * 80 + "\n")
            print(f"% --- Component LaTeX Table for {project_name} ({format_amplitude_title(lid_value)}) ---")
            print(component_latex_table)
            print("\n" + "-" * 80 + "\n")

def generate_time_to_fp64_level_table_best_proxy(
        entity="lokious-wageningen-uinversity",
        projects=None,
        metrics=("rRMSE",),
        max_seed=5,
        target_mode="mean_multiplier",
        target_multiplier=1.0,
        min_success_rate=0.80,
        table_stat="mean",
        group=None,
):
    """
    Fetch already logged W&B runs and generate a compact time-to-FP64-level table.
    FP64 and dynamci need to be saved in one project so it can be used as benchmark.
    Otherwise, use generate_time_to_manual_rrmse_table_by_strategy and manuually set the target rRMSE.

    """

    import wandb
    import pandas as pd
    import numpy as np

    if table_stat not in {"median", "mean"}:
        raise ValueError("table_stat must be either 'median' or 'mean'.")

    api = wandb.Api()

    if projects is None:
        projects = [
            # "pinn_irradiance_dynamic_precision",
            # "pinn_wave_dynamic_precision_MLP",
            # "pinn_reaction_dynamic_precision_MLP",
            # "pinn_convection_dynamic_precision_MLP",
            # "pinn_allen_cahn_dynamic_precision_MLP",
            # "pinn_ns2dc_dynamic_precision_PINN",
            # "pinn_beam_dynamic_precision_plotMLP",
            # "pinn_ns2dcg_dynamic_precisionPINN"
            # "pinn_convection_adaptive_weights"
            # "PINN_heat_dynamic_precision_MLP",
        ]

    project_equation_map = {
        "pinn_convection_dynamic_precision_plotMLP": "Convection",
        "pinn_convection_dynamic_precision_MLP": "Convection",
        "pinn_convection_dynamic_precision_3layerMLP": "Convection",

        "pinn_reaction_dynamic_precision_MLP": "Reaction",
        "pinn_reaction_dynamic_precision_3layerMLP": "Reaction",

        "pinn_wave_dynamic_precision_MLP": "Wave",
        "pinn_wave_dynamic_precision_3layerMLP": "Wave",

        "pinn_allen_cahn_dynamic_precision_MLP": "Allen",
        "pinn_allen_cahn_dynamic_precision_3layer": "Allen",
        "pinn_allen_cahn_dynamic_precision_3layerMLP": "Allen",

        "pinn_irradiance_dynamic_precision": "Irradiance",
        "PINN_heat_dynamic_precision_MLP": "Heat",
        "pinn_ns2dc_dynamic_precision_PINN": "NS2d-C",
        "pinn_beam_dynamic_precision_plotMLP": "Beam",
        "pinn_beam_dynamic_precision": "Beam",

    }
    metric_names = tuple(dict.fromkeys(metrics))
    group_col = str(group) if group is not None else None

    def safe_tag(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

    def get_group_value(config, summary, run, project, group_name):
        if group_name is None:
            return None

        group_key = str(group_name).lower()
        for source in (config or {}, summary or {}):
            for key, value in source.items():
                if str(key).lower() == group_key:
                    return value

        if group_key == "diagnostic_interval":
            texts = [project, getattr(run, "name", ""), getattr(run, "display_name", "")]
            patterns = [
                r"diagnostic[_-]?interval[_=-]?(\d+)",
                r"interval[_-]?(\d+)",
                r"(\d+)[_-]?step",
            ]
            for text in texts:
                for pattern in patterns:
                    match = re.search(pattern, str(text), flags=re.IGNORECASE)
                    if match:
                        return int(match.group(1))

        return "N/A"

    def infer_equation(project):
        if project in project_equation_map:
            return project_equation_map[project]
        parts = project.split("_")
        return parts[1].replace("-", " ").title() if len(parts) > 1 else project

    def infer_model(project):
        model = project.split("_")[-1]
        if model == "3layer":
            return "3layerMLP"
        if model == "precision":
            return "PINN"
        if model == "plotMLP":
            return "MLP"
        return model

    def clean_strategy(raw):
        raw = str(raw)
        low = raw.lower()

        if "switching" in low:
            return "skip"
        if raw == "fp64_curvature" or ("fp64" in low and "dynamic" not in low):
            return "fp64"
        if raw == "fp32_curvature" or ("fp32" in low and "dynamic" not in low):
            return "fp32"
        if raw in ["dynamic", "dynamic_precision_lbfgs", "dynamic precision lbfgs"]:
            return "dynamic precision"
        if "dynamic" in low or "curvature" in low:
            return "dynamic precision"

        return raw.replace("_", " ")

    def canonical_strategy(strategy):
        s = str(strategy).lower()
        if "dynamic" in s or "curvature" in s:
            return "dynamic"
        if "fp64" in s:
            return "fp64"
        if "fp32" in s:
            return "fp32"
        return str(strategy)

    def get_summary_metric(summary, metric):
        value = summary.get(metric)
        if value is None:
            value = summary.get(f"final_{metric}")
        return value

    def to_finite_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def round_for_target_comparison(value):
        try:
            value = float(value)
        except Exception:
            return np.nan
        if not np.isfinite(value):
            return np.nan
        return float(f"{value:.2e}")

    def find_component_metrics(summary, metric):
        suffix = f"_{metric}"
        components = []
        blocked_prefixes = (
            "mean_", "std_", "median_", "q25_", "q75_", "min_", "max_",
            "Mean_", "Std_", "Median_", "Q25_", "Q75_", "Min_", "Max_",
        )

        for key in summary.keys():
            name = str(key)
            if name.startswith("_"):
                continue
            candidate = name[len("final_"):] if name.startswith("final_") else name
            if candidate == metric or not candidate.endswith(suffix):
                continue
            if candidate.startswith(blocked_prefixes):
                continue
            if candidate not in components:
                components.append(candidate)

        return tuple(sorted(components))

    def get_metric_value_and_components(summary, metric):
        value = to_finite_float(get_summary_metric(summary, metric))
        if value is not None:
            return value, ()

        component_metrics = find_component_metrics(summary, metric)
        component_values = [
            to_finite_float(get_summary_metric(summary, component_metric))
            for component_metric in component_metrics
        ]
        component_values = [value for value in component_values if value is not None]
        if component_values:
            return float(np.mean(component_values)), component_metrics

        return None, ()

    def get_proxy_value(config, strategy):
        if canonical_strategy(strategy) != "dynamic":
            return "N/A"
        value = config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None))
        if value is None:
            value = 2.0
        try:
            return float(value)
        except Exception:
            return str(value)

    def fetch_metric_history(run, metric, component_metrics=()):
        def scan_keys(keys):
            try:
                rows = list(run.scan_history(keys=keys, page_size=10000))
                return pd.DataFrame(rows)
            except Exception as exc:
                print(f"scan_history failed for run {run.name}: {exc}")
                try:
                    return run.history(keys=keys, samples=100000)
                except Exception as exc2:
                    print(f"history fallback failed for run {run.name}: {exc2}")
                    return pd.DataFrame()

        def prepare_direct_history(hist):
            if hist.empty or "_runtime" not in hist.columns or metric not in hist.columns:
                return pd.DataFrame()

            cols = ["_runtime", metric]
            if "_step" in hist.columns:
                cols = ["_step", "_runtime", metric]

            hist = hist[cols].dropna().copy()
            hist["_runtime"] = pd.to_numeric(hist["_runtime"], errors="coerce")
            hist[metric] = pd.to_numeric(hist[metric], errors="coerce")
            hist = hist.dropna(subset=["_runtime", metric])
            hist = hist.sort_values("_runtime").reset_index(drop=True)
            return hist

        hist = prepare_direct_history(scan_keys(["_step", "_runtime", metric]))
        if not hist.empty:
            return hist

        component_metrics = tuple(component_metrics or ())
        if not component_metrics:
            return pd.DataFrame()

        component_hist = scan_keys(["_step", "_runtime", *component_metrics])
        if component_hist.empty or "_runtime" not in component_hist.columns:
            return pd.DataFrame()

        available_components = [
            component_metric for component_metric in component_metrics
            if component_metric in component_hist.columns
        ]
        if not available_components:
            return pd.DataFrame()

        cols = ["_runtime", *available_components]
        if "_step" in component_hist.columns:
            cols = ["_step", "_runtime", *available_components]

        component_hist = component_hist[cols].copy()
        component_hist["_runtime"] = pd.to_numeric(component_hist["_runtime"], errors="coerce")
        for component_metric in available_components:
            component_hist[component_metric] = pd.to_numeric(
                component_hist[component_metric],
                errors="coerce",
            )

        component_hist[metric] = component_hist[available_components].mean(axis=1, skipna=False)
        keep_cols = ["_runtime", metric]
        if "_step" in component_hist.columns:
            keep_cols = ["_step", "_runtime", metric]

        component_hist = component_hist[keep_cols].dropna(subset=["_runtime", metric])
        component_hist = component_hist.sort_values("_runtime").reset_index(drop=True)
        return component_hist

    def compute_target(fp64_values):
        vals = np.asarray(fp64_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        median = float(np.median(vals))
        q75 = float(np.quantile(vals, 0.75))
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        if target_mode in ["mean_multiplier", "mean_10pct", "mean_25pct"]:
            target = target_multiplier * mean
        elif target_mode == "q75":
            target = q75
        elif target_mode == "median_10pct":
            target = target_multiplier * median
        elif target_mode == "q75_or_10pct_median":
            target = max(q75, target_multiplier * median)
        elif target_mode == "mean_plus_sd":
            target = mean + std
        elif target_mode == "median":
            target = median
        else:
            raise ValueError(
                "Unsupported target_mode. Use one of: "
                "'mean_multiplier', 'mean_10pct', 'mean_25pct', "
                "'q75', 'median', 'median_10pct', "
                "'q75_or_10pct_median', 'mean_plus_sd'."
            )

        return float(target), median, q75, mean, std

    def finite_values(series):
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals

    def mean_std(series):
        vals = finite_values(series)
        if len(vals) == 0:
            return np.nan, np.nan
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    def median_value(series):
        vals = finite_values(series)
        if len(vals) == 0:
            return np.nan
        return float(np.median(vals))

    def median_or_inf(series):
        val = median_value(series)
        if pd.isna(val):
            return np.inf
        return val

    print(f"Fetching runs from entity: {entity}")

    base_rows = []

    for project in projects:
        runs = api.runs(path=f"{entity}/{project}")
        print(f"Found {len(runs)} runs in project '{project}'")

        equation = infer_equation(project)
        model = infer_model(project)

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            config = run.config or {}
            summary = run.summary or {}

            seed = config.get("seed")
            if seed is None:
                continue

            try:
                seed = int(seed)
            except Exception:
                continue

            if seed > max_seed:
                continue

            raw_strategy = config.get("strategy", "N/A")
            strategy = clean_strategy(raw_strategy)
            if strategy == "skip":
                continue

            pair_strategy = canonical_strategy(strategy)
            if pair_strategy not in ["dynamic", "fp32", "fp64"]:
                continue

            final_time = summary.get("training_time_seconds")
            if final_time is None:
                continue

            row = {
                "Project": project,
                "Equation": equation,
                "Model": model,
                "Seed": seed,
                "Raw_Strategy": raw_strategy,
                "Strategy": strategy,
                "Pair_Strategy": pair_strategy,
                "LOG_PROXY_LOW": get_proxy_value(config, strategy),
                "Training Time (s)": float(final_time),
                "wandb_run": run,
            }
            if group_col is not None:
                row[group_col] = get_group_value(config, summary, run, project, group_col)

            has_metric = False
            for metric in metric_names:
                value, component_metrics = get_metric_value_and_components(summary, metric)
                if value is not None:
                    row[f"Final_{metric}"] = value
                    row[f"Components_{metric}"] = component_metrics
                    has_metric = True
                else:
                    row[f"Final_{metric}"] = np.nan
                    row[f"Components_{metric}"] = ()

            if has_metric:
                base_rows.append(row)

    if not base_rows:
        print("No usable W&B runs found.")
        return None, None, None, None, None

    base_df = pd.DataFrame(base_rows)
    if group_col is not None:
        base_df[group_col] = base_df[group_col].fillna("N/A")
    target_group_cols = ["Equation", "Model"] if group_col is None else [group_col, "Equation", "Model"]
    proxy_group_cols = (["Equation", "Model", "Metric"] if group_col is None
                        else [group_col, "Equation", "Model", "Metric"])
    summary_group_cols = (["Equation", "Model", "Metric", "Pair_Strategy"] if group_col is None
                          else [group_col, "Equation", "Model", "Metric", "Pair_Strategy"])

    all_metric_rows = []

    for metric in metric_names:
        fp64_final = base_df[
            (base_df["Pair_Strategy"] == "fp64")
            & base_df[f"Final_{metric}"].notna()
        ]

        target_rows = []
        for group_key, sub in fp64_final.groupby(target_group_cols, observed=True):
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            key_context = dict(zip(target_group_cols, key_values))
            target, median, q75, mean, std = compute_target(sub[f"Final_{metric}"].values)
            if not np.isfinite(target):
                continue

            full_time_vals = finite_values(sub["Training Time (s)"])
            full_time_mean = float(np.mean(full_time_vals)) if len(full_time_vals) > 0 else np.nan
            full_time_median = float(np.median(full_time_vals)) if len(full_time_vals) > 0 else np.nan
            full_time_std = float(np.std(full_time_vals, ddof=1)) if len(full_time_vals) > 1 else 0.0

            target_rows.append({
                **key_context,
                "Metric": metric,
                "Target": target,
                "FP64_Median": median,
                "FP64_Q75": q75,
                "FP64_Mean": mean,
                "FP64_Std": std,
                "FP64_average_Mean": full_time_mean,
                "FP64_average_Median": full_time_median,
                "FP64_average_Std": full_time_std,
                "FP64_n": int(sub["Seed"].nunique()),
            })

        target_df = pd.DataFrame(target_rows)
        if target_df.empty:
            print(f"No FP64 target could be computed for metric {metric}.")
            continue

        metric_df = pd.merge(base_df, target_df, on=target_group_cols, how="inner")

        for _, row in metric_df.iterrows():
            run = row["wandb_run"]
            target = row["Target"]
            final_error = row[f"Final_{metric}"]
            final_time = row["Training Time (s)"]
            raw_components = row.get(f"Components_{metric}", ())
            if not isinstance(raw_components, (tuple, list)):
                raw_components = ()
            component_metrics = tuple(raw_components)
            metric_source = "component_mean" if component_metrics else "logged"

            hist = fetch_metric_history(run, metric, component_metrics)

            reached = False
            time_to_target = np.nan
            source = "not_reached"
            target_cmp = round_for_target_comparison(target)

            if not hist.empty:
                metric_cmp = hist[metric].map(round_for_target_comparison)
                hit = hist[metric_cmp <= target_cmp]
                if not hit.empty:
                    reached = True
                    time_to_target = float(hit.iloc[0]["_runtime"])
                    source = "component_history" if component_metrics else "history"

            if (
                not reached
                and pd.notna(final_error)
                and round_for_target_comparison(final_error) <= target_cmp
            ):
                reached = True
                time_to_target = float(final_time)
                source = "component_summary_fallback" if component_metrics else "summary_fallback"

            metric_row = {
                "Project": row["Project"],
                "Equation": row["Equation"],
                "Model": row["Model"],
                "Metric": metric,
                "Target": float(target),
                "FP64_Median": float(row["FP64_Median"]),
                "FP64_Q75": float(row["FP64_Q75"]),
                "FP64_Mean": float(row["FP64_Mean"]),
                "FP64_Std": float(row["FP64_Std"]),
                "FP64_average_Mean": float(row["FP64_average_Mean"]),
                "FP64_average_Median": float(row["FP64_average_Median"]),
                "FP64_average_Std": float(row["FP64_average_Std"]),
                "Seed": int(row["Seed"]),
                "Strategy": row["Strategy"],
                "Pair_Strategy": row["Pair_Strategy"],
                "LOG_PROXY_LOW": row["LOG_PROXY_LOW"],
                "Metric_Source": metric_source,
                "Component_Metrics": ", ".join(component_metrics) if component_metrics else metric,
                "Final_Error": float(final_error) if pd.notna(final_error) else np.nan,
                "Final_Time": float(final_time),
                "Reached_Target": bool(reached),
                "Time_to_Target": time_to_target,
                "Source": source,
            }
            if group_col is not None:
                metric_row[group_col] = row[group_col]
            all_metric_rows.append(metric_row)

    all_runs = pd.DataFrame(all_metric_rows)
    if all_runs.empty:
        print("No metric histories or final summaries could be converted to time-to-target.")
        return base_df, None, None, None, None

    # Compare all time-to-target values with mean FP64 training time
    all_runs["Relative_Time_to_FP64_average"] = (
        all_runs["Time_to_Target"] / all_runs["FP64_average_Mean"]
    )
    all_runs["Speedup_vs_FP64_average"] = (
        all_runs["FP64_average_Mean"] / all_runs["Time_to_Target"]
    )

    fail_mask = ~all_runs["Reached_Target"]
    all_runs.loc[
        fail_mask,
        ["Relative_Time_to_FP64_average", "Speedup_vs_FP64_average"]
    ] = np.nan


    selected_proxy_rows = []
    dynamic_rows = all_runs[all_runs["Pair_Strategy"] == "dynamic"].copy()

    for group_key, sub in dynamic_rows.groupby(proxy_group_cols, observed=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_context = dict(zip(proxy_group_cols, key_values))
        proxy_summaries = []

        for proxy, g in sub.groupby("LOG_PROXY_LOW", observed=True):
            n_total = int(g["Seed"].nunique())
            n_success = int(g[g["Reached_Target"]]["Seed"].nunique())
            success_rate = n_success / max(n_total, 1)

            rel_average_median = median_or_inf(g.loc[g["Reached_Target"], "Relative_Time_to_FP64_average"])
            ttt_median = median_or_inf(g.loc[g["Reached_Target"], "Time_to_Target"])
            final_error_median = median_or_inf(g["Final_Error"])

            proxy_summaries.append({
                **key_context,
                "LOG_PROXY_LOW": proxy,
                "n_total": n_total,
                "n_success": n_success,
                "success_rate": success_rate,
                "median_relative_fp64_average_time": rel_average_median,
                "median_time_to_target": ttt_median,
                "median_final_error": final_error_median,
            })

        proxy_df = pd.DataFrame(proxy_summaries)
        if proxy_df.empty:
            continue

        eligible = proxy_df[proxy_df["success_rate"] >= min_success_rate].copy()

        if not eligible.empty:
            eligible = eligible.sort_values(
                by=["median_relative_fp64_average_time", "median_time_to_target", "median_final_error"],
                ascending=[True, True, True],
            )
            selected = eligible.iloc[0].copy()
            selected["selection_rule"] = (
                f"success_rate >= {min_success_rate:.2f}, then lowest median time-to-target relative to average FP64 time"
            )
        else:
            proxy_df = proxy_df.sort_values(
                by=["success_rate", "median_final_error", "median_relative_fp64_average_time"],
                ascending=[False, True, True],
            )
            selected = proxy_df.iloc[0].copy()
            selected["selection_rule"] = (
                "no proxy reached minimum success rate; selected highest success "
                "then lowest final error"
            )

        selected_proxy_rows.append(selected)

    selected_proxy = pd.DataFrame(selected_proxy_rows)

    if selected_proxy.empty:
        print("No dynamic proxy setting could be selected.")
        selected_runs = all_runs[all_runs["Pair_Strategy"].isin(["fp32", "fp64"])].copy()
    else:
        selected_proxy_merge_cols = proxy_group_cols + ["LOG_PROXY_LOW"]
        dyn_selected = pd.merge(
            dynamic_rows,
            selected_proxy[selected_proxy_merge_cols],
            on=selected_proxy_merge_cols,
            how="inner",
        )

        non_dynamic = all_runs[all_runs["Pair_Strategy"].isin(["fp32", "fp64"])].copy()
        selected_runs = pd.concat([non_dynamic, dyn_selected], ignore_index=True)


    summary_rows = []

    for group_key, sub in selected_runs.groupby(summary_group_cols, observed=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_context = dict(zip(summary_group_cols, key_values))
        eq = key_context["Equation"]
        model = key_context["Model"]
        metric = key_context["Metric"]
        strategy = key_context["Pair_Strategy"]
        n_total = int(sub["Seed"].nunique())
        n_success = int(sub[sub["Reached_Target"]]["Seed"].nunique())
        reached = sub[sub["Reached_Target"]].copy()

        t_mean, t_std = mean_std(reached["Time_to_Target"])
        t_median = median_value(reached["Time_to_Target"])
        final_mean, final_std = mean_std(sub["Final_Error"])
        final_median = median_value(sub["Final_Error"])

        fp64_average_mean = float(sub["FP64_average_Mean"].iloc[0])
        fp64_average_median = float(sub["FP64_average_Median"].iloc[0])
        fp64_average_std = float(sub["FP64_average_Std"].iloc[0])

        rel_mean = t_mean / fp64_average_mean if pd.notna(t_mean) and fp64_average_mean > 0 else np.nan
        rel_median = t_median / fp64_average_mean if pd.notna(t_median) and fp64_average_mean > 0 else np.nan

        speedup_mean = fp64_average_mean / t_mean if pd.notna(t_mean) and t_mean > 0 else np.nan
        speedup_median = fp64_average_mean / t_median if pd.notna(t_median) and t_median > 0 else np.nan

        selected_proxy_value = "N/A"
        selection_rule = "N/A"
        if strategy == "dynamic" and not selected_proxy.empty:
            proxy_mask = (
                (selected_proxy["Equation"] == eq)
                & (selected_proxy["Model"] == model)
                & (selected_proxy["Metric"] == metric)
            )
            if group_col is not None:
                proxy_mask &= selected_proxy[group_col] == key_context[group_col]
            proxy_match = selected_proxy[proxy_mask]
            if not proxy_match.empty:
                selected_proxy_value = proxy_match.iloc[0]["LOG_PROXY_LOW"]
                selection_rule = proxy_match.iloc[0]["selection_rule"]

        summary_row = {
            "Metric": metric,
            "Equation": eq,
            "Model": model,
            "Strategy": strategy,
            "Selected_LOG_PROXY_LOW": selected_proxy_value,
            "Selection_Rule": selection_rule,
            "Target": float(sub["Target"].iloc[0]),
            "FP64_Mean": float(sub["FP64_Mean"].iloc[0]),
            "FP64_Std": float(sub["FP64_Std"].iloc[0]),
            "FP64_average_Mean": fp64_average_mean,
            "FP64_average_Median": fp64_average_median,
            "FP64_average_Std": fp64_average_std,
            "Metric_Source": ", ".join(sorted(sub["Metric_Source"].dropna().unique())),
            "Component_Metrics": ", ".join(sorted(sub["Component_Metrics"].dropna().unique())),
            "n": n_total,
            "Success_N": n_success,
            "Success": f"{n_success}/{n_total}",
            "Mean_Time_to_Target": t_mean,
            "Std_Time_to_Target": t_std,
            "Median_Time_to_Target": t_median,
            "Mean_Relative_Time_to_FP64_average": rel_mean,
            "Median_Relative_Time_to_FP64_average": rel_median,
            "Mean_Speedup_vs_FP64_average": speedup_mean,
            "Median_Speedup_vs_FP64_average": speedup_median,
            "Mean_Final_Error": final_mean,
            "Std_Final_Error": final_std,
            "Median_Final_Error": final_median,
        }
        if group_col is not None:
            summary_row[group_col] = key_context[group_col]
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)

    base_order = [
        "Convection", "Reaction", "Wave", "Allen", "Irradiance",
        "Heat"
    ]
    extra_equations = [
        eq for eq in summary["Equation"].dropna().unique().tolist()
        if eq not in base_order
    ]
    ordered_equations = base_order + sorted(extra_equations)
    ordered_strategies = ["dynamic", "fp32", "fp64"]

    summary["Equation"] = pd.Categorical(summary["Equation"], categories=ordered_equations, ordered=True)
    summary["Metric"] = pd.Categorical(summary["Metric"], categories=metric_names, ordered=True)
    summary["Strategy"] = pd.Categorical(summary["Strategy"], categories=ordered_strategies, ordered=True)
    summary_sort_cols = ["Metric", "Equation", "Model", "Strategy"]
    if group_col is not None:
        summary_sort_cols = [group_col] + summary_sort_cols
    summary = summary.sort_values(summary_sort_cols).reset_index(drop=True)
    summary["Equation"] = summary["Equation"].astype(str)
    summary["Metric"] = summary["Metric"].astype(str)
    summary["Strategy"] = summary["Strategy"].astype(str)


    display_decimals = 2

    def fmt_num(x, precision=display_decimals):
        if pd.isna(x) or not np.isfinite(x):
            return "NR"
        return f"{x:.{precision}f}"

    def fmt_target(x):
        if pd.isna(x) or not np.isfinite(x):
            return "-"
        return f"{x:.2e}"

    def fmt_speedup(x):
        if pd.isna(x) or not np.isfinite(x):
            return "NR"
        return rf"\({x:.{display_decimals}f}\times\)"

    def fmt_metric(metric):
        return str(metric).replace("_", r"\_")

    def get_summary_row(metric, eq, model, strategy, group_value=None):
        match = summary[
            (summary["Metric"] == metric)
            & (summary["Equation"] == eq)
            & (summary["Model"] == model)
            & (summary["Strategy"] == strategy)
        ]
        if group_col is not None:
            match = match[match[group_col] == group_value]
        if match.empty:
            return None
        return match.iloc[0]

    table_rows = []
    show_metric_column = len(metric_names) > 1
    table_group_cols = ["Metric", "Equation", "Model"] if group_col is None else [group_col, "Metric", "Equation", "Model"]

    for group_key, table_group in summary.groupby(table_group_cols, observed=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_context = dict(zip(table_group_cols, key_values))
        group_value = key_context.get(group_col) if group_col is not None else None
        metric = key_context["Metric"]
        eq = key_context["Equation"]
        model = key_context["Model"]

        dyn_row = get_summary_row(metric, eq, model, "dynamic", group_value)
        if dyn_row is None:
            continue

        fp64_row = get_summary_row(metric, eq, model, "fp64", group_value)
        fp32_row = get_summary_row(metric, eq, model, "fp32", group_value)

        fp64_average_time = dyn_row["FP64_average_Mean"]
        if fp64_row is not None and pd.notna(fp64_row["FP64_average_Mean"]):
            fp64_average_time = fp64_row["FP64_average_Mean"]

        fp64_success = "0/0" if fp64_row is None else fp64_row["Success"]
        if table_stat == "median":
            dyn_ttt = dyn_row["Median_Time_to_Target"]
            dyn_speedup = dyn_row["Median_Speedup_vs_FP64_average"]
        else:
            dyn_ttt = dyn_row["Mean_Time_to_Target"]
            dyn_speedup = dyn_row["Mean_Speedup_vs_FP64_average"]

        fp32_success = "0/0" if fp32_row is None else fp32_row["Success"]

        table_row = {
            "Equation": eq,
            "Target rRMSE": fmt_target(dyn_row["Target"]),
            "FP64 average time (s)": fmt_num(fp64_average_time),
            r"FP64 success": fp64_success,
            r"FP32 success": fp32_success,
            r"Dynamic success": dyn_row["Success"],
            r"Dynamic to target(s)": fmt_num(dyn_ttt),
            r"Speedup": fmt_speedup(dyn_speedup),
        }
        if group_col is not None:
            table_row = {group_col: group_value, **table_row}
        if show_metric_column:
            if group_col is not None:
                table_row = {group_col: group_value, "Metric": fmt_metric(metric), **table_row}
            else:
                table_row = {"Metric": fmt_metric(metric), **table_row}

        table_rows.append(table_row)

    latex_df = pd.DataFrame(table_rows)

    caption = (
        rf"Time-to-target comparison using average FP64 result as target. "
        rf"When overall rRMSE is unavailable, matching component rRMSE metrics "
        rf"are averaged."
    )
    if group_col is not None:
        caption = rf"{caption} Results are grouped by {group_col}."

    leading_text_cols = 1 + int(show_metric_column) + int(group_col is not None)

    latex_str = latex_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=(
            "tab:time_to_fp64_level_best_proxy_fp64_average"
            if group_col is None
            else f"tab:time_to_fp64_level_best_proxy_fp64_average_by_{safe_tag(group_col)}"
        ),
        column_format="l" * leading_text_cols + "r" * (len(latex_df.columns) - leading_text_cols),
    )

    print(latex_str)

    return all_runs, selected_runs, selected_proxy, summary, latex_str

def generate_time_to_manual_rrmse_table_by_strategy(
        target_rrmse,
        project="pinn_convection_adaptive_weights",
        entity="lokious-wageningen-uinversity",
        metric="rRMSE",
        max_seed=5,
        table_stat="mean",
        group_dynamic_by_proxy=True,
        group=None,
        output_path=None,
        caption=None,
        label=None,
):
    """
    Generate a LaTeX table for one or more projects using a manually chosen target.

    For each finished W&B run:
        time_to_target = first logged _runtime where metric <= target_rrmse

    """

    import wandb
    import pandas as pd
    import numpy as np

    target_rrmse = float(target_rrmse)
    if not np.isfinite(target_rrmse):
        raise ValueError("target_rrmse must be a finite number.")

    if table_stat not in {"median", "mean"}:
        raise ValueError("table_stat must be either 'median' or 'mean'.")

    projects = list(project) if isinstance(project, (list, tuple, set)) else [project]
    project_label = ", ".join(projects)
    group_col = str(group) if group is not None else None

    def safe_tag(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

    def get_group_value(config, summary, run, project_name, group_name):
        if group_name is None:
            return None

        group_key = str(group_name).lower()
        for source in (config or {}, summary or {}):
            for key, value in source.items():
                if str(key).lower() == group_key:
                    return value

        if group_key == "diagnostic_interval":
            texts = [project_name, getattr(run, "name", ""), getattr(run, "display_name", "")]
            patterns = [
                r"diagnostic[_-]?interval[_=-]?(\d+)",
                r"interval[_-]?(\d+)",
                r"(\d+)[_-]?step",
            ]
            for text in texts:
                for pattern in patterns:
                    match = re.search(pattern, str(text), flags=re.IGNORECASE)
                    if match:
                        return int(match.group(1))

        return "N/A"

    def to_finite_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def canonical_strategy(strategy):
        s = str(strategy).lower()
        if "dynamic" in s or "curvature" in s:
            return "dynamic"
        if "fp64" in s:
            return "fp64"
        if "fp32" in s:
            return "fp32"
        return str(strategy).replace("_", " ").lower()

    def clean_strategy(raw_strategy):
        raw = str(raw_strategy)
        low = raw.lower()

        if "switching" in low:
            return None, None
        if raw == "fp64_curvature" or ("fp64" in low and "dynamic" not in low):
            return "fp64", "fp64"
        if raw == "fp32_curvature" or ("fp32" in low and "dynamic" not in low):
            return "fp32", "fp32"
        if "dynamic" in low or "curvature" in low:
            return "dynamic precision", "dynamic"

        clean = raw.replace("_", " ")
        return clean, canonical_strategy(clean)

    def get_proxy_value(config, pair_strategy):
        if pair_strategy != "dynamic":
            return "N/A"
        value = config.get("LOG_PROXY_LOW", config.get("log_proxy_low", None))
        if value is None:
            value = 2.0
        return value

    def format_proxy_value(value):
        try:
            return f"{float(value):g}"
        except Exception:
            return str(value)

    def display_strategy(strategy, pair_strategy, proxy):
        if pair_strategy == "dynamic" and group_dynamic_by_proxy:
            return f"{strategy} (Proxy {format_proxy_value(proxy)})"
        return strategy

    def get_summary_metric(summary, metric_name):
        value = summary.get(metric_name)
        if value is None:
            value = summary.get(f"final_{metric_name}")
        return value

    def find_component_metrics(summary, metric_name):
        suffix = f"_{metric_name}"
        components = []
        blocked_prefixes = (
            "mean_", "std_", "median_", "q25_", "q75_", "min_", "max_",
            "Mean_", "Std_", "Median_", "Q25_", "Q75_", "Min_", "Max_",
        )

        for key in summary.keys():
            name = str(key)
            if name.startswith("_"):
                continue
            candidate = name[len("final_"):] if name.startswith("final_") else name
            if candidate == metric_name or not candidate.endswith(suffix):
                continue
            if candidate.startswith(blocked_prefixes):
                continue
            if candidate not in components:
                components.append(candidate)

        return tuple(sorted(components))

    def get_metric_value_and_components(summary, metric_name):
        value = to_finite_float(get_summary_metric(summary, metric_name))
        if value is not None:
            return value, ()

        component_metrics = find_component_metrics(summary, metric_name)
        component_values = [
            to_finite_float(get_summary_metric(summary, component_metric))
            for component_metric in component_metrics
        ]
        component_values = [value for value in component_values if value is not None]
        if component_values:
            return float(np.mean(component_values)), component_metrics

        return None, ()

    def fetch_metric_history(run, metric_name, component_metrics=()):
        def scan_keys(keys):
            try:
                rows = list(run.scan_history(keys=keys, page_size=10000))
                return pd.DataFrame(rows)
            except Exception as exc:
                print(f"scan_history failed for run {run.name}: {exc}")
                try:
                    return run.history(keys=keys, samples=100000)
                except Exception as exc2:
                    print(f"history fallback failed for run {run.name}: {exc2}")
                    return pd.DataFrame()

        def prepare_direct_history(hist):
            if hist.empty or "_runtime" not in hist.columns or metric_name not in hist.columns:
                return pd.DataFrame()

            cols = ["_runtime", metric_name]
            if "_step" in hist.columns:
                cols = ["_step", "_runtime", metric_name]

            hist = hist[cols].dropna().copy()
            hist["_runtime"] = pd.to_numeric(hist["_runtime"], errors="coerce")
            hist[metric_name] = pd.to_numeric(hist[metric_name], errors="coerce")
            hist = hist.dropna(subset=["_runtime", metric_name])
            hist = hist.sort_values("_runtime").reset_index(drop=True)
            return hist

        hist = prepare_direct_history(scan_keys(["_step", "_runtime", metric_name]))
        if not hist.empty:
            return hist

        component_metrics = tuple(component_metrics or ())
        if not component_metrics:
            return pd.DataFrame()

        component_hist = scan_keys(["_step", "_runtime", *component_metrics])
        if component_hist.empty or "_runtime" not in component_hist.columns:
            return pd.DataFrame()

        available_components = [
            component_metric for component_metric in component_metrics
            if component_metric in component_hist.columns
        ]
        if not available_components:
            return pd.DataFrame()

        cols = ["_runtime", *available_components]
        if "_step" in component_hist.columns:
            cols = ["_step", "_runtime", *available_components]

        component_hist = component_hist[cols].copy()
        component_hist["_runtime"] = pd.to_numeric(component_hist["_runtime"], errors="coerce")
        for component_metric in available_components:
            component_hist[component_metric] = pd.to_numeric(
                component_hist[component_metric],
                errors="coerce",
            )

        component_hist[metric_name] = component_hist[available_components].mean(axis=1, skipna=False)
        keep_cols = ["_runtime", metric_name]
        if "_step" in component_hist.columns:
            keep_cols = ["_step", "_runtime", metric_name]

        component_hist = component_hist[keep_cols].dropna(subset=["_runtime", metric_name])
        component_hist = component_hist.sort_values("_runtime").reset_index(drop=True)
        return component_hist

    def finite_values(series):
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals

    def mean_std(series):
        vals = finite_values(series)
        if len(vals) == 0:
            return np.nan, np.nan
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    def median_value(series):
        vals = finite_values(series)
        if len(vals) == 0:
            return np.nan
        return float(np.median(vals))

    def fmt_num(value, precision=2):
        if pd.isna(value) or not np.isfinite(value):
            return "NR"
        return f"{value:.{precision}f}"

    def fmt_target(value):
        if pd.isna(value) or not np.isfinite(value):
            return "-"
        return f"{value:.4e}"

    def fmt_success_rate(success_count, total_count):
        if total_count <= 0:
            return "0.0\\% (0/0)"
        return f"{100.0 * success_count / total_count:.1f}\\% ({success_count}/{total_count})"

    print(f"Fetching runs from {entity}/{project_label} with target {metric} <= {target_rrmse:.4e}")

    api = wandb.Api()

    rows = []

    for project_name in projects:
        runs = api.runs(path=f"{entity}/{project_name}")
        print(f"Found {len(runs)} runs in project '{project_name}'")

        for run in runs:
            if run.state != "finished" or not run.summary:
                continue

            config = run.config or {}
            summary = run.summary or {}

            seed = config.get("seed")
            if seed is None:
                continue

            try:
                seed = int(seed)
            except Exception:
                continue

            if max_seed is not None and seed > max_seed:
                continue

            raw_strategy = config.get("strategy", "N/A")
            strategy, pair_strategy = clean_strategy(raw_strategy)
            if strategy is None:
                continue

            final_time = to_finite_float(summary.get("training_time_seconds"))
            if final_time is None:
                continue

            final_error, component_metrics = get_metric_value_and_components(summary, metric)
            if final_error is None:
                continue

            proxy = get_proxy_value(config, pair_strategy)
            hist = fetch_metric_history(run, metric, component_metrics)

            reached = False
            time_to_target = np.nan
            source = "not_reached"

            if not hist.empty:
                hit = hist[hist[metric] <= target_rrmse]
                if not hit.empty:
                    reached = True
                    time_to_target = float(hit.iloc[0]["_runtime"])
                    source = "component_history" if component_metrics else "history"

            if not reached and final_error <= target_rrmse:
                reached = True
                time_to_target = final_time
                source = "component_summary_fallback" if component_metrics else "summary_fallback"

            run_row = {
                "Project": project_name,
                "Run": run.name,
                "Seed": seed,
                "Raw_Strategy": raw_strategy,
                "Strategy": strategy,
                "Pair_Strategy": pair_strategy,
                "Table_Strategy": display_strategy(strategy, pair_strategy, proxy),
                "LOG_PROXY_LOW": proxy,
                "Target": target_rrmse,
                "Metric": metric,
                "Metric_Source": "component_mean" if component_metrics else "logged",
                "Component_Metrics": ", ".join(component_metrics) if component_metrics else metric,
                "Final_Error": final_error,
                "Final_Time": final_time,
                "Reached_Target": reached,
                "Time_to_Target": time_to_target,
                "Source": source,
            }
            if group_col is not None:
                run_row[group_col] = get_group_value(config, summary, run, project_name, group_col)
            rows.append(run_row)

    if not rows:
        print("No usable W&B runs found.")
        return None, None, None

    all_runs = pd.DataFrame(rows)
    if group_col is not None:
        all_runs[group_col] = all_runs[group_col].fillna("N/A")

    summary_rows = []
    summary_group_cols = "Table_Strategy" if group_col is None else [group_col, "Table_Strategy"]
    for group_key, sub in all_runs.groupby(summary_group_cols, observed=True):
        if group_col is None:
            group_value = None
            strategy = group_key
        else:
            group_value, strategy = group_key
        n_total = int(sub["Seed"].nunique())
        n_success = int(sub[sub["Reached_Target"]]["Seed"].nunique())
        reached = sub[sub["Reached_Target"]].copy()

        mean_time, std_time = mean_std(reached["Time_to_Target"])
        median_time = median_value(reached["Time_to_Target"])
        mean_final, std_final = mean_std(sub["Final_Error"])
        median_final = median_value(sub["Final_Error"])

        pair_strategy = str(sub["Pair_Strategy"].iloc[0])
        proxy_values = sorted(
            {format_proxy_value(value) for value in sub["LOG_PROXY_LOW"].tolist() if str(value) != "N/A"}
        )

        summary_row = {
            "Strategy": strategy,
            "Pair_Strategy": pair_strategy,
            "LOG_PROXY_LOW": ", ".join(proxy_values) if proxy_values else "N/A",
            "Target": target_rrmse,
            "n": n_total,
            "Success_N": n_success,
            "Success_Rate": n_success / max(n_total, 1),
            "Success": f"{n_success}/{n_total}",
            "Mean_Time_to_Target": mean_time,
            "Std_Time_to_Target": std_time,
            "Median_Time_to_Target": median_time,
            "Mean_Final_Error": mean_final,
            "Std_Final_Error": std_final,
            "Median_Final_Error": median_final,
        }
        if group_col is not None:
            summary_row[group_col] = group_value
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)

    strategy_order = {"fp64": 0, "fp32": 1, "dynamic": 2}
    summary["_sort_pair"] = summary["Pair_Strategy"].map(strategy_order).fillna(99)
    summary["_sort_proxy"] = pd.to_numeric(summary["LOG_PROXY_LOW"], errors="coerce")
    sort_cols = ["_sort_pair", "_sort_proxy", "Strategy"]
    if group_col is not None:
        sort_cols = [group_col] + sort_cols
    summary = summary.sort_values(
        sort_cols,
        na_position="last",
    ).drop(columns=["_sort_pair", "_sort_proxy"]).reset_index(drop=True)

    time_col = "Median_Time_to_Target" if table_stat == "median" else "Mean_Time_to_Target"
    target_col = "Target rRMSE" if metric == "rRMSE" else f"Target {metric}".replace("_", r"\_")

    table_rows = []
    for _, row in summary.iterrows():
        table_row = {
            "Strategy": row["Strategy"],
            "Training time (s)": fmt_num(row[time_col], precision=2),
            "Success rate": fmt_success_rate(int(row["Success_N"]), int(row["n"])),
            target_col: fmt_target(row["Target"]),
        }
        if group_col is not None:
            table_row = {group_col: row[group_col], **table_row}
        table_rows.append(table_row)

    latex_df = pd.DataFrame(table_rows)

    if caption is None:
        caption = (
            rf"Time-to-target comparison for {project_label} using a manually chosen "
            rf"target {target_col} of {fmt_target(target_rrmse)}. Training time "
            rf"is the {table_stat} first wall-clock time among successful runs."
        )
        if group_col is not None:
            caption = rf"{caption} Results are grouped by {group_col}."

    if label is None:
        label = f"tab:manual_target_rrmse_by_strategy_{safe_tag(project_label)}"
        if group_col is not None:
            label = f"{label}_by_{safe_tag(group_col)}"

    leading_text_cols = 1 + int(group_col is not None)

    latex_str = latex_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" * leading_text_cols + "r" * (len(latex_df.columns) - leading_text_cols),
    )

    print(latex_str)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to {output_path}")

    return all_runs, summary, latex_str

def plot_bar_metrics(
        project_name="pinn_convection_dynamic_precision_MLP",
        entity="lokious-wageningen-uinversity",
        num_seeds=5
):
    """
    This is to create figure of motivation plot
    """
    import wandb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    print(f"Fetching runs from {entity}/{project_name}...")
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project_name}")

    data = []

    # Extract data from W&B
    for run in runs:
        if run.state != "finished" or not run.summary:
            continue

        config = run.config
        seed = config.get("seed")

        if seed is None or seed >= num_seeds:
            continue

        time_s = run.summary.get("training_time_seconds")
        rrmse = run.summary.get("rRMSE")

        if time_s is None or rrmse is None:
            continue

        raw_strategy = config.get("strategy", "N/A")
        if "switching" in raw_strategy.lower():
            continue

        # Standardize strategy names and exclusively grab FP32/FP64
        if raw_strategy == "fp64_curvature" or (
                "fp64" in raw_strategy.lower() and "dynamic" not in raw_strategy.lower()):
            strategy = "FP64"
        elif raw_strategy == "fp32_curvature" or (
                "fp32" in raw_strategy.lower() and "dynamic" not in raw_strategy.lower()):
            strategy = "FP32"
        else:
            continue

        data.append({
            "Seed": seed,
            "Strategy": strategy,
            "rRMSE": rrmse,
            "Training Time (s)": time_s
        })

    df = pd.DataFrame(data)

    if df.empty:
        print("No valid FP32 or FP64 runs found for the requested seeds and project!")
        return

    target_strategies = ["FP32", "FP64"]

    target_strategies = [s for s in target_strategies if s in df["Strategy"].unique()]

    # Calculate Mean and Std Deviation
    agg_df = df.groupby("Strategy").agg(
        rRMSE_mean=("rRMSE", "mean"),
        rRMSE_std=("rRMSE", "std"),
        Time_mean=("Training Time (s)", "mean"),
        Time_std=("Training Time (s)", "std")
    ).reindex(target_strategies).dropna()

    print("\n--- Aggregated Data (FP32 & FP64) ---")
    print(agg_df)


    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    x = np.arange(len(target_strategies))
    width = 0.35


    strategy_colors = {'FP32': '#7F7F7F', 'FP64': '#1F77B4'}
    bar_colors = [strategy_colors.get(strat, '#333333') for strat in target_strategies]

    # Plot rRMSE on the left axis (Shifted slightly left, Solid fill)
    rects1 = ax1.bar(x - width / 2, agg_df["rRMSE_mean"], width, yerr=agg_df["rRMSE_std"],
                     color=bar_colors, capsize=8, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

    # Plot Training Time on the right axis (Shifted slightly right, Hatched fill)
    rects2 = ax2.bar(x + width / 2, agg_df["Time_mean"], width, yerr=agg_df["Time_std"],
                     color=bar_colors, capsize=8, alpha=0.85,
                     edgecolor='black', linewidth=1.5, hatch='//')


    ax1.set_ylabel('rRMSE', fontweight='bold')
    ax2.set_ylabel('Training Time (s)', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(target_strategies, fontweight='bold')

    ax1.set_title(f"rRMSE & Training Time Comparison\n({project_name})")
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    legend_elements = [
        Patch(facecolor='#7F7F7F', edgecolor='black', alpha=0.85, label='FP32'),
        Patch(facecolor='#1F77B4', edgecolor='black', alpha=0.85, label='FP64'),
        Patch(facecolor='white', edgecolor='black', label='rRMSE'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Training Time')
    ]

    ax1.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    plt.tight_layout()
    filename = f"{project_name}_dual_axis_FP32_FP64.svg"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved dual-axis grouped bar plot to {filename}")
    plt.close(fig)


def plot_example_dynamic_log_proxy(
        project_name="pinn_convection_dynamic_precision_MLP",
        entity="lokious-wageningen-uinversity",
        threshold=2.5,
        slope_threshold=0.03
):
    """
    Plot an example dynamic run's smoothed log curvature proxy over training steps"""
    import wandb
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    print(f"Fetching an example dynamic run from {entity}/{project_name}...")
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project_name}")

    target_run = None

    for run in runs:
        if run.state != "finished":
            continue

        strat = run.config.get("strategy", "").lower()
        if "dynamic" in strat or "curvature" in strat:
            target_run = run
            break

    if not target_run:
        print("No dynamic run found to plot.")
        return

    history = target_run.history(keys=["_step", "smoothed_log_proxy"], pandas=True)
    if history.empty or "smoothed_log_proxy" not in history.columns:
        print("Run doesn't contain 'smoothed_log_proxy' history.")
        return

    history = history.dropna().sort_values("_step")
    x = history["_step"].values
    y = history["smoothed_log_proxy"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = []
    for segment in segments:
        x1, y1 = segment[0]
        x2, y2 = segment[1]

        # Calculate slope (prevent division by zero just in case)
        dx = x2 - x1 if (x2 - x1) != 0 else 1e-9
        slope = abs((y2 - y1) / dx)

        if y1 >= threshold or y2 >= threshold or slope > slope_threshold:
            colors.append('#1F77B4')
        else:
            colors.append('#7F7F7F')

    lc = LineCollection(segments, colors=colors, linewidths=2.5)
    ax.add_collection(lc)

    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Threshold ({threshold})')

    ax.plot([], [], color='#1F77B4', linewidth=2.5,
            label=f'FP64' )
    ax.plot([], [], color='#7F7F7F', linewidth=2.5,
            label='FP32')

    ax.autoscale()

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\tilde{z}_t$ (Log Curvature Proxy)")
    ax.set_title(f"Dynamic Proxy Behavior vs Threshold\n(Run Seed: {target_run.config.get('seed', 'N/A')})")

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

    plt.tight_layout()
    filename = f"example_dynamic_proxy_threshold_{project_name}.svg"
    plt.savefig(filename, dpi=300)
    print(f"Saved example plot to {filename}")
    plt.close(fig)

def main():
    plt.rcParams.update({
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "legend.title_fontsize": 22
    })
    generate_results_summary()
    plot_bar_metrics()
    plot_example_dynamic_log_proxy()

    #Result table in Section 4.6
    analyze_hyperparameter_groups_ns2d_c()
    analyze_hyperparameter_groups_PDE_system()
    
    #Figure 9
    plot_single_seed_from_project(
        project="pinn_wave_dynamic_precision_PINNMamba",
        seed=0
    )
    generate_time_to_fp64_level_table_best_proxy()

    #Appendix table D.1
    generate_time_to_manual_rrmse_table_by_strategy(
        target_rrmse=1.08e-2,# the target value from fp64
        project="pinn_convection_dynamic_precision_5step_MLP",
        group="DIAGNOSTIC_INTERVAL",
    )
    generate_time_to_manual_rrmse_table_by_strategy(
        target_rrmse=5.78e-2,
        project="pinn_reaction_dynamic_precision_5step_MLP",
        group="DIAGNOSTIC_INTERVAL",
    )
    generate_time_to_manual_rrmse_table_by_strategy(
        target_rrmse=1.47e-2,
        project="pinn_wave_dynamic_precision_5step_MLP",
        group="DIAGNOSTIC_INTERVAL",
    )
    generate_time_to_manual_rrmse_table_by_strategy(
        target_rrmse=5.03e-2,
        project="pinn_allen_cahn_dynamic_precision_5step_MLP",
        group="DIAGNOSTIC_INTERVAL",
    )
if __name__ == "__main__":
    main()
