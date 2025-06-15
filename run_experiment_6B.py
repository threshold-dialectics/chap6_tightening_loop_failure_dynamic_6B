import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, chi2_contingency
import os
import json
import io
from contextlib import redirect_stdout

def smooth(y, box_pts):
    """A simple moving average smoothing function."""
    if box_pts <= 1:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # Handle boundary effects from convolution
    y_smooth[:box_pts//2] = y[:box_pts//2]
    y_smooth[-box_pts//2:] = y[-box_pts//2:]
    return y_smooth

class TDSystem:
    """A simple system to demonstrate the 'Tightening Loop' failure dynamic."""
    def __init__(self, params):
        self.params = params
        # Initial State
        self.g_lever = params.get('g0', 1.0)
        self.beta_lever = params.get('beta0', 1.0)
        self.fcrit_lever = params.get('fcrit0', 10.0)
        self.strain = params.get('strain0', 0.5)
        
        # System Architectural Parameters
        self.w1 = params.get('w1', 0.2)
        self.w2 = params.get('w2', 0.2)
        self.w3 = params.get('w3', 0.6)
        
        self.theta_t = self._update_theta_t()
        self.history = []

    def _update_theta_t(self):
        """Calculates the Tolerance Sheet value."""
        return (self.g_lever**self.w1 * 
                self.beta_lever**self.w2 * 
                self.fcrit_lever**self.w3)

    def _calculate_beta_cost(self):
        """Calculates the direct cost of policy precision using a convex model."""
        return self.params['kappa_beta'] * (self.beta_lever ** self.params['phi_2'])

    def step(self, t):
        """Advances the simulation by one time step."""
        p = self.params
        
        # 1. Environmental pressure increases strain after an initial period
        if t > p['strain_start_time']:
            self.strain += p['strain_increase_rate']
        
        # 2. Implement the maladaptive "doubling down" policy
        if self.strain > p['beta_response_threshold']:
            self.beta_lever += p['beta_response_rate'] * (self.strain - p['beta_response_threshold'])

        # 3. Calculate costs and deplete energetic slack (F_crit)
        beta_cost = self._calculate_beta_cost()
        total_cost = p['base_cost'] + beta_cost
        self.fcrit_lever -= total_cost
        self.fcrit_lever = max(self.fcrit_lever, 0.01) # Prevent negative values

        # 4. Update Tolerance and check for collapse
        self.theta_t = self._update_theta_t()
        is_collapsed = self.strain > self.theta_t

        # 5. Record current state
        self.history.append({
            'time': t,
            'g_lever': self.g_lever,
            'beta_lever': self.beta_lever,
            'fcrit_lever': self.fcrit_lever,
            'strain': self.strain,
            'theta_t': self.theta_t,
            'is_collapsed': is_collapsed
        })
        return is_collapsed

    def run_simulation(self, steps):
        """Runs the simulation until collapse or for a max number of steps."""
        for t in range(steps):
            if self.step(t):
                # Run a few more steps post-collapse to show the breach clearly
                for _ in range(3):
                    if t < steps - 1:
                        t += 1
                        self.step(t)
                break
        return pd.DataFrame(self.history)

def perform_analysis(df, baseline_window, precollapse_window):
    """Calculates diagnostics and performs statistical tests."""
    # --- Calculate Diagnostics ---
    smoothing_window = 10
    df['beta_smooth'] = smooth(df['beta_lever'], smoothing_window)
    df['fcrit_smooth'] = smooth(df['fcrit_lever'], smoothing_window)

    df['beta_dot'] = np.gradient(df['beta_smooth'])
    df['fcrit_dot'] = np.gradient(df['fcrit_smooth'])
    
    df['speed_index'] = np.sqrt(df['beta_dot']**2 + df['fcrit_dot']**2)
    
    correlation_window = 20
    df['couple_index'] = df['beta_dot'].rolling(window=correlation_window).corr(df['fcrit_dot'])
    df.fillna(0, inplace=True)

    # --- Statistical Analysis ---
    baseline_df = df[(df['time'] >= baseline_window[0]) & (df['time'] <= baseline_window[1])]
    precollapse_df = df[(df['time'] >= precollapse_window[0]) & (df['time'] <= precollapse_window[1])]

    # H1: Speed Index comparison
    speed_u, speed_p = mannwhitneyu(baseline_df['speed_index'], precollapse_df['speed_index'], alternative='less')
    
    # H2: Couple Index comparison
    couple_u, couple_p = mannwhitneyu(baseline_df['couple_index'], precollapse_df['couple_index'], alternative='greater')

    # H3: Tightening Loop Prevalence
    df['is_tightening'] = (df['beta_dot'] > 0) & (df['fcrit_dot'] < 0)
    contingency_table = pd.crosstab(
        df.loc[baseline_df.index.union(precollapse_df.index), 'time'].apply(lambda x: 'precollapse' if x in precollapse_df.index else 'baseline'),
        df.loc[baseline_df.index.union(precollapse_df.index), 'is_tightening']
    )
    chi2, chi2_p, dof, _ = chi2_contingency(contingency_table)

    stats_results = {
        'baseline_mean_speed': baseline_df['speed_index'].mean(),
        'precollapse_mean_speed': precollapse_df['speed_index'].mean(),
        'speed_u': speed_u, 'speed_p': speed_p,
        'baseline_mean_couple': baseline_df['couple_index'].mean(),
        'precollapse_mean_couple': precollapse_df['couple_index'].mean(),
        'couple_u': couple_u, 'couple_p': couple_p,
        'contingency_table': contingency_table,
        'chi2': chi2, 'chi2_p': chi2_p
    }
    
    return df, stats_results

def print_statistical_summary(stats):
    """Prints a formatted summary of the statistical results."""
    print("--- Statistical Analysis Summary ---")
    print("\nComparing Baseline vs. Pre-Collapse Windows:")
    
    # Speed Index
    print(f"\nH1: Speed Index increases pre-collapse")
    print(f"  - Mean Speed (Baseline):    {stats['baseline_mean_speed']:.4f}")
    print(f"  - Mean Speed (Pre-Collapse): {stats['precollapse_mean_speed']:.4f}")
    print(f"  - Mann-Whitney U test: U={stats['speed_u']:.1f}, p={stats['speed_p']:.4f}")
    if stats['speed_p'] < 0.05:
        print("  - Result: Significant increase in speed pre-collapse.")
    else:
        print("  - Result: No significant increase detected.")

    # Couple Index
    print(f"\nH2: Couple Index becomes more negative pre-collapse")
    print(f"  - Mean Couple (Baseline):    {stats['baseline_mean_couple']:.4f}")
    print(f"  - Mean Couple (Pre-Collapse): {stats['precollapse_mean_couple']:.4f}")
    print(f"  - Mann-Whitney U test: U={stats['couple_u']:.1f}, p={stats['couple_p']:.4f}")
    if stats['couple_p'] < 0.05:
        print("  - Result: Significant negative shift in coupling pre-collapse.")
    else:
        print("  - Result: No significant shift detected.")

    # Tightening Loop Prevalence
    print(f"\nH3: 'Tightening Loop' dynamic is more prevalent pre-collapse")
    print("Contingency Table:")
    print(stats['contingency_table'])
    print(f"\n  - Chi-squared test: chi2({stats['contingency_table'].shape[0]-1})={stats['chi2']:.2f}, p={stats['chi2_p']:.4g}")
    if stats['chi2_p'] < 0.05:
        print("  - Result: Significant association between the pre-collapse window and the tightening dynamic.")
    else:
        print("  - Result: No significant association detected.")

def plot_results(df, stats, save_path=None):
    """Creates the multi-panel visualization of the simulation results."""
    # Double font sizes for publication clarity
    title_fontsize = 28
    label_fontsize = 22
    tick_fontsize = 18
    legend_fontsize = 16

    fig = plt.figure(figsize=(18, 14))  # Bigger figure
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle('Experiment 6B: Emergence of the "Tightening Loop" Dynamic', fontsize=title_fontsize, y=0.96)

    # Corrected legends
    rigidity_label = "Rigidity (beta_p) Up"
    slack_label = "Energetic Slack (Fcrit_p) Down"

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['time'], df['beta_lever'], color='blue', label=rigidity_label)
    ax1.plot(df['time'], df['fcrit_lever'], color='red', label=slack_label)
    ax1.set_title('Lever Proxies vs. Time', fontsize=label_fontsize)
    ax1.set_xlabel('Time Step', fontsize=label_fontsize)
    ax1.set_ylabel('Lever Value', fontsize=label_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['time'], df['strain'], color='black', label='Strain (avg_DeltaP_p)')
    ax2.plot(df['time'], df['theta_t'], color='purple', linestyle='--', label='Tolerance (ThetaT_p)')
    ax2.fill_between(df['time'], df['strain'], df['theta_t'], where=df['theta_t'] > df['strain'],
                     color='green', alpha=0.2, label='Safety Margin')
    ax2.set_title('Strain vs. Tolerance', fontsize=label_fontsize)
    ax2.set_xlabel('Time Step', fontsize=label_fontsize)
    ax2.set_ylabel('Value', fontsize=label_fontsize)
    ax2.tick_params(labelsize=tick_fontsize)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Panel 3
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['time'], df['speed_index'], color='orange', label='Speed Index')
    ax3.plot(df['time'], df['couple_index'], color='purple', label='Couple Index')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_title('TD Diagnostics vs. Time', fontsize=label_fontsize)
    ax3.set_xlabel('Time Step', fontsize=label_fontsize)
    ax3.set_ylabel('Index Value', fontsize=label_fontsize)
    ax3.set_ylim(-1.1, 1.1)
    ax3.tick_params(labelsize=tick_fontsize)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=legend_fontsize)

    # Panel 4
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(df['speed_index'], df['couple_index'], c=df['time'], cmap='viridis', s=30, alpha=0.7)
    ax4.set_title('Trajectory on Diagnostic Plane', fontsize=label_fontsize)
    ax4.set_xlabel('Speed Index', fontsize=label_fontsize)
    ax4.set_ylabel('Couple Index', fontsize=label_fontsize)
    ax4.set_xlim(left=-0.005)
    ax4.set_ylim(-1.1, 1.1)
    ax4.axhline(0, color='gray', linestyle='--')
    ax4.tick_params(labelsize=tick_fontsize)
    ax4.grid(True, linestyle='--', alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax4)
    cbar.set_label('Time Step', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Add collapse line
    collapse_time = df[df['is_collapsed']].time.iloc[0] if not df[df['is_collapsed']].empty else None
    if collapse_time:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(collapse_time, color='r', linestyle=':', linewidth=2.5, label='Collapse')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', ncol=2, fontsize=legend_fontsize)
    ax2.legend().set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=350)  # <- DPI boost here
    plt.show()


if __name__ == '__main__':
    # --- Simulation Setup ---
    simulation_params = {
        'beta0': 1.0, 'fcrit0': 10.0, 'strain0': 0.5,
        'strain_start_time': 20, 'strain_increase_rate': 0.015,
        'beta_response_threshold': 0.6, 'beta_response_rate': 0.02,
        'w1': 0.2, 'w2': 0.15, 'w3': 0.65, # High reliance on Fcrit, low on Beta
        'kappa_beta': 0.01, 'phi_2': 1.8,  # Convex cost for beta
        'base_cost': 0.02
    }
    
    # --- Analysis Window Definitions ---
    BASELINE_WINDOW = (0, 30)
    PRECOLLAPSE_WINDOW_DURATION = 30

    # --- Execution ---
    system = TDSystem(simulation_params)
    results_df = system.run_simulation(steps=200)

    if not results_df.empty:
        # Define the pre-collapse window based on actual collapse time
        collapse_time = results_df[results_df['is_collapsed']].time.iloc[0] if not results_df[results_df['is_collapsed']].empty else len(results_df)
        precollapse_start = max(0, collapse_time - PRECOLLAPSE_WINDOW_DURATION)
        PRECOLLAPSE_WINDOW = (precollapse_start, collapse_time - 1)
        
        # Perform analysis
        analyzed_df, stats_results = perform_analysis(results_df, BASELINE_WINDOW, PRECOLLAPSE_WINDOW)
        
        # Ensure result directory exists
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)

        # Capture console output from the statistical summary
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_statistical_summary(stats_results)
        console_output = buffer.getvalue()
        print(console_output)

        # Save console output to JSON file
        summary_path = os.path.join(result_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({"console_output": console_output}, f, indent=2)

        # Plot results and save the figure
        figure_path = os.path.join(result_dir, "experiment_6B_results.png")
        plot_results(analyzed_df, stats_results, save_path=figure_path)
