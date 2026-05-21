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
            ax.set_title(f'Log Proxy vs Step - {equation}', fontsize=21)
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


def generate_results_summary():
    import wandb
    import pandas as pd
    import numpy as np

    api = wandb.Api()

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

        "pinn_wave_dynamic_precision_3layerMLP",
        "pinn_reaction_dynamic_precision_3layerMLP",
        "pinn_convection_dynamic_precision_3layerMLP",
        "pinn_allen_cahn_dynamic_precision_3layerMLP",
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

            raw_runs_data.append({
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
             })

    if not raw_runs_data:
        print("No valid runs found with seeds between 0 and 4!")
        return

    df = pd.DataFrame(raw_runs_data)

    df["Training Time (s)"] = pd.to_numeric(df["Training Time (s)"], errors='coerce')
    df["rRMSE"] = pd.to_numeric(df["rRMSE"], errors='coerce')
    df["rMAE"] = pd.to_numeric(df["rMAE"], errors='coerce')
    df.dropna(subset=["Training Time (s)", "rRMSE", "rMAE"], inplace=True)

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
    df.to_csv("all_runs_data.csv", index=False)

    group_perf = df.groupby(['Equation', 'Model', 'Strategy', 'LOG_PROXY_LOW'])['Training Time (s)'].mean().reset_index()
    print("Mean training times by group:")
    print(group_perf)

    best_idx = group_perf.groupby(['Equation', 'Model', 'Strategy'])['Training Time (s)'].idxmin()
    best_groups = group_perf.loc[best_idx, ['Equation', 'Model', 'Strategy', 'LOG_PROXY_LOW']]

    df = pd.merge(df, best_groups, on=['Equation', 'Model', 'Strategy', 'LOG_PROXY_LOW'], how='inner')

    best_run_ids = set(df['wandb_run'].apply(lambda r: r.id))
    all_runs_data = [r for r in all_runs_data if r['wandb_run'].id in best_run_ids]

    print(f"\nFiltered down to {len(df)} optimal runs across {len(best_groups)} unique strategy groups.")

    print("\n" + "=" * 80)
    print("--- SELECTED OPTIMAL HYPERPARAMETERS (Used for Plots & Tables) ---")
    print("=" * 80)
    for (eq, model), group in best_groups.groupby(['Equation', 'Model']):
        print(f"\n[{model}] Equation: {eq}")
        for _, row in group.iterrows():
            print(f"  - Strategy: {row['Strategy']}")
            print(f"    LOG_PROXY_LOW: {row['LOG_PROXY_LOW']}")
    print("\n" + "=" * 80 + "\n")

    baseline_strats = {}
    for eq, model in df[['Equation', 'Model']].drop_duplicates().values:
        mask = (df['Equation'] == eq) & (df['Model'] == model)
        eq_model_strats = df[mask]['Strategy'].unique()
        fp64_strats = [s for s in eq_model_strats if 'fp64' in s.lower() and 'dynamic' not in s.lower()]
        if fp64_strats:
            baseline_strats[(eq, model)] = fp64_strats[0]

    agg_df = df.groupby(["Equation", "Model", "Strategy"]).agg(
        Mean_Time=('Training Time (s)', 'mean'), Std_Time=('Training Time (s)', 'std'),
        Mean_rRMSE=('rRMSE', 'mean'), Std_rRMSE=('rRMSE', 'std'),
        Mean_rMAE=('rMAE', 'mean'), Std_rMAE=('rMAE', 'std'),
        Run_Count=('Seed', 'count')
    ).reset_index()
    print(agg_df)

    agg_df = agg_df[agg_df['Run_Count'] >= 2]
    ordered_equations = ["Convection", "Reaction", "Wave", "Allen", "Irradiance"]
    agg_df["Equation"] = pd.Categorical(agg_df["Equation"], categories=ordered_equations, ordered=True)
    agg_df.sort_values(by=["Equation", "Model", "Strategy"], inplace=True)
    agg_df.reset_index(drop=True, inplace=True)

    agg_df['Mean_Speedup'] = np.nan
    fp64_baselines = {}
    for eq, model in agg_df[['Equation', 'Model']].drop_duplicates().values:
        mask = (agg_df['Equation'] == eq) & (agg_df['Model'] == model)
        eq_df = agg_df[mask]
        fp64_mask = eq_df['Strategy'].str.contains('fp64', case=False, na=False) & ~eq_df['Strategy'].str.contains('dynamic', case=False, na=False)
        if any(fp64_mask):
            fp64_run = eq_df[fp64_mask].iloc[0]
            fp64_baselines[(eq, model)] = {'rRMSE': fp64_run['Mean_rRMSE'], 'rMAE': fp64_run['Mean_rMAE'], 'Time': fp64_run['Mean_Time']}
            agg_df.loc[mask, 'Mean_Speedup'] = fp64_run['Mean_Time'] / agg_df.loc[mask, 'Mean_Time']

    agg_df['is_green'] = False
    for idx, row in agg_df.iterrows():
        key = (row['Equation'], row['Model'])
        if 'fp64' not in row['Strategy'].lower() and key in fp64_baselines:
            baseline = fp64_baselines[key]
            is_rrmse_green = pd.notnull(row['Mean_rRMSE']) and row['Mean_rRMSE'] <= 10 * baseline['rRMSE']
            is_rmae_green = pd.notnull(row['Mean_rMAE']) and row['Mean_rMAE'] <= 10 * baseline['rMAE']
            if is_rrmse_green and is_rmae_green:
                agg_df.loc[idx, 'is_green'] = True

    green_df = agg_df[agg_df['is_green']].copy()
    min_time_indices = green_df.groupby(["Equation", "Model"], observed=True)["Mean_Time"].idxmin()
    min_rrmse_indices = green_df.groupby(["Equation", "Model"], observed=True)["Mean_rRMSE"].idxmin()
    min_rmae_indices = green_df.groupby(["Equation", "Model"], observed=True)["Mean_rMAE"].idxmin()
    max_speedup_indices = green_df.groupby(["Equation", "Model"], observed=True)["Mean_Speedup"].idxmax()

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
        key = (eq_name, model_name)
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
    for line in lines:
        if '\\midrule' in line: in_body = True
        if '\\bottomrule' in line: in_body = False
        if in_body and '&' in line and row_idx > 0 and agg_df['Equation'].iloc[row_idx] != agg_df['Equation'].iloc[row_idx - 1]:
            processed_lines.append('\\hline')
        processed_lines.append(line)
        if in_body and '&' in line: row_idx += 1
    print('\n'.join(processed_lines))

    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    def get_seed_speedup(row):
        key = (row['Equation'], row['Model'])
        if key in baseline_strats:
            b_strat = baseline_strats[key]
            match = df[(df['Equation'] == row['Equation']) & (df['Model'] == row['Model']) & (df['Seed'] == row['Seed']) & (df['Strategy'] == b_strat)]
            if not match.empty:
                baseline_time = match.iloc[0]['Training Time (s)']
                if row['Training Time (s)'] > 0:
                    return baseline_time / row['Training Time (s)']
        return np.nan

    df['Speedup_vs_FP64'] = df.apply(get_seed_speedup, axis=1)

    for model_name, model_df in df.groupby('Model'):
        print(f"\n--- Generating plots for model: {model_name} ---")

        model_run_ids = set(model_df['wandb_run'].apply(lambda r: r.id))
        model_all_runs_data = [r for r in all_runs_data if r['wandb_run'].id in model_run_ids]
        model_agg_df = agg_df[agg_df['Model'] == model_name]

        plot_training_time_vs_accuracy(model_df, model_name)
        plot_l2_vs_walltime(model_df, model_all_runs_data, model_name)
        plot_l2_vs_step(model_df, model_all_runs_data, model_name)
        plot_log_proxy_vs_step(model_df, model_all_runs_data, model_name)


    print("\nPlots generated successfully!")

def analyze_hyperparameter_groups():
    """
    Fetches all runs, groups them by project, equation, and strategy,
    and generates a LaTeX summary table for each project analyzing performance and convergence.
    """
    import wandb
    import pandas as pd

    api = wandb.Api()

    ENTITY = "lokious-wageningen-uinversity"
    PROJECTS = [
        # "pinn_wave_dynamic_precision_PINNMamba",
        # "pinn_reaction_dynamic_precision_PINNMamba",
        # "pinn_convection_dynamic_precision_PINNMamba",
        # "pinn_allen_cahn_dynamic_precision_PINNMamba",
        # "pinn_convection_dynamic_precision_PINNsFormer",
        # "pinn_reaction_dynamic_precision_PINNsFormer",
        "pinn_wave_dynamic_precision_PINNsFormer",
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
        Patch(facecolor='white', edgecolor='black', label='rRMSE (Left Axis)'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Training Time (Right Axis)')
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
    # plot_bar_metrics()
    # plot_example_dynamic_log_proxy()
    # analyze_hyperparameter_groups()
    # plot_single_seed_from_project(
    #     project="pinn_wave_dynamic_precision_PINNMamba",
    #     seed=0
    # )


if __name__ == "__main__":
    main()