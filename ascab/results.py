import os
import pickle
import argparse

import numpy as np
import pandas as pd

from ascab.utils.plot import plot_normalized_reward, plot_pesticide_use, plot_results, plot_risk, plot_use_vs_risk


def separate_underscore(string, index = 0):
    string = string.split('_')
    if 'pkl' in string[-1]:
        return string[index][:-4]
    else:
        return string[index]


def extract_metrics(results_dict):

    dict_extracted = {"Reward": {}, "Pesticide": {}, 'Pesticide_actions': {}, 'Precipitation': {}, 'Risk': {}}
    years_list = []

    for k, df in results_dict.items():

        if "Reward" in df.columns:
            df['Year'] = df['Date'].dt.year
            reward_per_year = df.groupby('Year')['Reward'].sum()
            for year, reward in reward_per_year.items():
                dict_extracted['Reward'].setdefault(year, []).append(reward)

        if "Pesticide" in df.columns:
            df['Year'] = df['Date'].dt.year
            pesticide_per_year = df.groupby('Year')['Pesticide'].sum()
            for year, pesticide in pesticide_per_year.items():
                dict_extracted['Pesticide'].setdefault(year, []).append(pesticide)

            pesticide_actions_per_year = df.groupby('Year')['Pesticide']
            for year, group in pesticide_actions_per_year:
                dict_extracted['Pesticide_actions'].setdefault(year, []).append(group)

        if "Precipitation" in df.columns:
            df['Year'] = df['Date'].dt.year
            precipitation_per_year = df.groupby('Year')['Precipitation']
            for year, prec in precipitation_per_year:
                dict_extracted['Precipitation'].setdefault(year, []).append(prec)

        if "Risk" in df.columns:
            df['Year'] = df['Date'].dt.year
            risk_per_year = df.groupby('Year')['Risk'].sum()
            for year, risk in risk_per_year.items():
                dict_extracted['Risk'].setdefault(year, []).append(risk)

        years_list.append(k)

    return dict_extracted

# Bootstrap function
def bootstrap_metrics(values, n_boot=10000):
    medians = []
    iqr_vals = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        medians.append(np.median(sample))
        q75, q25 = np.percentile(sample, [75, 25])
        iqr_vals.append(q75 - q25)
    return np.array(medians), np.array(iqr_vals)

def main(args):

    this_file_path = os.path.abspath(__file__)
    pkl_dir = os.path.join(os.path.dirname(this_file_path), 'results')
    pkl_dir_baselines = pkl_dir
    if args.trunc:
        pkl_dir = os.path.join(os.path.dirname(this_file_path), 'results', 'rppo')
    baseline_pickle_names = ['ceres.pkl', 'random.pkl', 'umbrella.pkl', 'zero.pkl']

    results_dict = {}
    baselines_dict = {}

    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl') and filename not in baseline_pickle_names:
            with open(os.path.join(pkl_dir, filename), 'rb') as f:
                # assumes filename is something like 'rl_agent_DQN_seed1.pkl'
                results_dict[separate_underscore(filename, -1)] = pickle.load(f)
    for filename in os.listdir(pkl_dir_baselines):
        if filename in baseline_pickle_names:
            with open(os.path.join(pkl_dir_baselines, filename), 'rb') as f:
                baselines_dict[filename[:-4]] = pickle.load(f)

    random_dict = baselines_dict.pop('random')

    random_extracted = extract_metrics(random_dict)
    dict_extracted = extract_metrics(results_dict)
    baselines_extracted = extract_metrics(baselines_dict)

    results = {
        "random": random_extracted,
        "baselines": baselines_extracted,
        "rl": dict_extracted,
    }

    baseline_names = ["Ceres", "Umbrella", "Zero"]


    for category in ["Reward", "Pesticide", "Risk"]:
        print(f"\n{category} stats per year:")
        for year in sorted(dict_extracted[category].keys()):
            values = dict_extracted[category][year]
            mean_val = np.mean(values)
            std_val = np.std(values)
            # median_val = np.median(values)
            # iqr_val = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            values_ran = random_extracted[category][year]
            mean_val_ran = np.mean(values_ran)
            std_val_ran = np.std(values_ran)
            print(f"Year {year}: RL mean = {mean_val:.3f}, std = {std_val:.4f}")
            print(f"           Random mean = {mean_val_ran:.3f}, std = {std_val_ran:.4f}")
            for i, baseline_name in enumerate(baseline_names):
                value = baselines_extracted[category][year][i]
                print(f"           {baseline_name} = {value:.3f}")

    plot_it = True
    if plot_it:
        plot_normalized_reward(dict_extracted, baselines_extracted, random_extracted, plot_type='bar')
        plot_pesticide_use(dict_extracted, baselines_extracted, random_extracted)
        plot_risk(dict_extracted, baselines_extracted, random_extracted)
        plot_use_vs_risk(dict_extracted, baselines_extracted, random_extracted)

        dict_to_plot = {"Zero":baselines_dict["zero"],
             "Umbrella":baselines_dict["umbrella"],
             "Ceres":baselines_dict["ceres"],
             "RL":results_dict[list(results_dict.keys())[1]],
             "Random":random_dict[next(iter(random_dict))],}
        # for k, v in dict_to_plot.items():
        for zoom in [True, False]:
            plot_results(
                dict_to_plot,
                variables=[
                    "Precipitation",
                     "AscosporeMaturation",
                     "Discharge",
                     "Pesticide",
                     "Risk",
                     "Action",
                ],
                save_path=os.path.join(pkl_dir),
                per_year=True,
                zoom=zoom,
            )

    statistics = False
    if statistics:
        # Define agent groups
        baseline_agents = ["Ceres", "Umbrella", "Zero"]
        seed_names = [
            'seed101871', 'seed104838', 'seed354986', 'seed427066',
            'seed486074', 'seed677211', 'seed683253', 'seed710178',
            'seed89331'
        ]

        for x in ['Reward', 'Pesticide', 'Risk']:

            print(f'{x} summary')

            # Remap baselines['Reward'] lists to dicts keyed by agent
            results['baselines'][x] = {
                year: dict(zip(baseline_agents, vals))
                for year, vals in results['baselines'][x].items()
            }

            # Remap rl['Reward'] lists to dicts keyed by seed
            results['rl'][x] = {
                year: dict(zip(seed_names, vals))
                for year, vals in results['rl'][x].items()
            }

            # Extract values
            ceres_vals = np.array([
                results['baselines'][x][yr]["Ceres"]
                for yr in sorted(results['baselines'][x])
            ])
            umb_vals = np.array([
                results['baselines'][x][yr]["Umbrella"]
                for yr in sorted(results['baselines'][x])
            ])
            rl_vals = np.concatenate([
                list(results['rl'][x][yr].values())
                for yr in sorted(results['rl'][x])
            ])

            # Bootstrap both medians and their difference in one loop
            n_boot = 10000
            diffs_ceres = np.empty(n_boot)
            diffs_umb = np.empty(n_boot)
            for i in range(n_boot):
                samp_c = np.random.choice(ceres_vals, size=len(ceres_vals), replace=True)
                samp_r = np.random.choice(rl_vals, size=len(rl_vals), replace=True)
                samp_u = np.random.choice(umb_vals, size=len(umb_vals), replace=True)
                diffs_ceres[i] = np.median(samp_c) - np.median(samp_r)
                diffs_umb[i] = np.median(samp_r) - np.median(samp_u)
            if x == 'Reward':
                p_one_sided_ceres = np.sum(diffs_ceres <= 0) / n_boot
                p_one_sided_umb = np.sum(diffs_umb <= 0) / n_boot
            elif x == 'Pesticide':
                p_one_sided_ceres = np.sum(diffs_ceres > 0) / n_boot
                p_one_sided_umb = np.sum(diffs_umb > 0) / n_boot

            # Summarize
            median_diff_ceres = np.median(diffs_ceres)
            iqr_ceres = np.percentile(diffs_ceres, 75) - np.percentile(diffs_ceres, 25)
            ci_lower_ceres, ci_upper_ceres = np.percentile(diffs_ceres, [2.5, 97.5])


            print("Bootstrapped Median Difference (Ceres - RL):", median_diff_ceres)
            print("IQR (Ceres - RL):", iqr_ceres)
            print("95% CI Ceres - RL:", (ci_lower_ceres, ci_upper_ceres))
            print("One sided p-value Ceres: ", p_one_sided_ceres)


            # Summarize
            median_diff_umb = np.median(diffs_umb)
            iqr_umb = np.percentile(diffs_umb, 75) - np.percentile(diffs_umb, 25)
            ci_lower_umb, ci_upper_umb = np.percentile(diffs_umb, [2.5, 97.5])

            print("Bootstrapped Median Difference (RL - Umbrella):", median_diff_umb)
            print("IQR (RL - Umbrella):", iqr_umb)
            print("95% CI RL - Umbrella:", (ci_lower_umb, ci_upper_umb))
            print("One sided p-value Umbrella: ", p_one_sided_umb)


            # Gather reward values for each agent
            data = {}

            # Baselines: one value per year per agent
            years_baselines = sorted(results['baselines'][x].keys())
            for agent in baseline_agents:
                data[agent] = [results['baselines'][x][year][agent] for year in years_baselines]

            # RL: one value per seed per year
            data["RL"] = []
            for year in sorted(results['rl'][x].keys()):
                data["RL"].extend(results['rl'][x][year].values())

            # Random: 15 runs per year
            data["Random"] = []
            for year in sorted(results['random'][x].keys()):
                data["Random"].extend(results['random'][x][year])


            # Compute bootstrap summaries
            summary = []
            for agent, vals in data.items():
                vals = np.array(vals)
                medians_bs, iqr_bs = bootstrap_metrics(vals)
                summary.append({
                    "Agent": agent,
                    "Bootstrapped Median": np.median(medians_bs),
                    "Bootstrapped IQR": np.median(iqr_bs),
                    "95% CI Lower": np.percentile(medians_bs, 2.5),
                    "95% CI Upper": np.percentile(medians_bs, 97.5),
                })

            df_summary = pd.DataFrame(summary)

            print(df_summary)
            print("\n")

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--trunc', action='store_true')
    arg = argparse.parse_args()
    main(arg)