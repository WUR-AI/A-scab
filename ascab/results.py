import os
import pickle
import argparse

import numpy as np
import pandas as pd

from ascab.utils.plot import (plot_normalized_reward,
                              plot_pesticide_use,
                              plot_results,
                              plot_risk,
                              plot_use_vs_risk,
                              plot_use_and_risk_bars,
                              plot_pesticide_vs_risk_pies,
                              plot_pesticide_vs_risk_donuts)


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
            pesticide_per_year = df.groupby('Year')['Action'].sum()
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
    pkl_dir = os.path.join(os.path.dirname(this_file_path), 'results', 'rppo')
    pkl_dir_baselines = pkl_dir
    # if args.trunc:
    #     pkl_dir = os.path.join(os.path.dirname(this_file_path), 'results', 'rppo')
    print(f"loading results from {pkl_dir}")
    baseline_pickle_names = ['1ceres.pkl', '3super_farmer.pkl', '4farmers_practice.pkl', '5random.pkl', '6zero.pkl']

    results_dict = {}
    baselines_dict = {}

    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl') and filename not in baseline_pickle_names:
            with open(os.path.join(pkl_dir, filename), 'rb') as f:
                # assumes filename is something like 'rl_agent_DQN_seed1.pkl'
                results_dict[separate_underscore(filename, -1)] = pickle.load(f)
    for filename in baseline_pickle_names:
        path = os.path.join(pkl_dir_baselines, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                baselines_dict[filename[:-4]] = pickle.load(f)

    random_dict = baselines_dict.pop('5random')

    random_extracted = extract_metrics(random_dict)
    dict_extracted = extract_metrics(results_dict)
    baselines_extracted = extract_metrics(baselines_dict)

    results = {
        "random": random_extracted,
        "baselines": baselines_extracted,
        "rl": dict_extracted,
    }

    baseline_names = ["Ceres", "Super Farmer", "Farmer's Practice", "Zero"]


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
        plot_normalized_reward(dict_extracted, baselines_extracted, random_extracted, save_path=pkl_dir, use_umbrella=True)
        # plot_pesticide_use(dict_extracted, baselines_extracted, random_extracted, pareto_line=True, save_path=pkl_dir)
        # plot_risk(dict_extracted, baselines_extracted, random_extracted, avg_line=True, save_path=pkl_dir)
        # plot_use_vs_risk(dict_extracted, baselines_extracted, random_extracted)
        plot_use_and_risk_bars(dict_extracted, baselines_extracted, random_extracted, save_path=pkl_dir, use_umbrella=True)
        # plot_pesticide_vs_risk_pies(dict_extracted, baselines_extracted, random_extracted)
        # plot_pesticide_vs_risk_donuts(dict_extracted, baselines_extracted, random_extracted)

    plot_years = True
    if plot_years:
        dict_to_plot = {
            "Ceres": baselines_dict["1ceres"],
            "Super Farmer": baselines_dict["3super_farmer"],
            "RL": results_dict[list(results_dict.keys())[0]],
            "Farmer's Practice":baselines_dict["4farmers_practice"],
            "Random": list(random_dict.values())[1], # random_dict[next(iter(random_dict))],
            "Zero":baselines_dict["6zero"],}
        # for k, v in dict_to_plot.items():
        for zoom in [True]:
            plot_results(
                dict_to_plot,
                save_path=os.path.join(pkl_dir),
                per_year=True,
                zoom=zoom,
                stacked=True if zoom else False,
            )

    statistics = True
    if statistics:
        # Define agent groups
        baseline_agents = ["Ceres", "SPractice", "FPractice", "Zero"]
        seed_names = [
            'seed101871', 'seed104838', 'seed354986', 'seed427066',
            'seed486074', 'seed677211', 'seed683253', 'seed710178',
            'seed89331',
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
                results['baselines'][x][yr]["SPractice"]
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
            elif x == "Risk":
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

    tab3_latex(pkl_dir, pkl_dir_baselines)


def _tab3_default_results_dir():
    this_file_path = os.path.abspath(__file__)
    return os.path.join(os.path.dirname(this_file_path), 'results', 'rppo')


def _tab3_load_pickles(
        pkl_dir=None,
        pkl_dir_baselines=None,
        baseline_pickle_names=None,
):
    if pkl_dir is None:
        pkl_dir = _tab3_default_results_dir()
    if pkl_dir_baselines is None:
        pkl_dir_baselines = pkl_dir
    if baseline_pickle_names is None:
        baseline_pickle_names = ['1ceres.pkl', '3super_farmer.pkl', '4farmers_practice.pkl', '5random.pkl', '6zero.pkl']

    results_dict = {}
    baselines_dict = {}

    for filename in sorted(os.listdir(pkl_dir)):
        if filename.endswith('.pkl') and filename not in baseline_pickle_names:
            with open(os.path.join(pkl_dir, filename), 'rb') as f:
                results_dict[separate_underscore(filename, -1)] = pickle.load(f)

    for filename in baseline_pickle_names:
        path = os.path.join(pkl_dir_baselines, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                baselines_dict[filename[:-4]] = pickle.load(f)

    random_dict = baselines_dict.pop('5random')
    return results_dict, baselines_dict, random_dict


def _tab3_bootstrap_median_ci(values, n_boot=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    values = np.asarray(values)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[i] = np.median(sample)

    return {
        "median": np.median(medians),
        "ci_lower": np.percentile(medians, 2.5),
        "ci_upper": np.percentile(medians, 97.5),
    }


def _tab3_bootstrap_differences(ceres_vals, rl_vals, super_farmer_vals, metric, n_boot=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    ceres_vals = np.asarray(ceres_vals)
    rl_vals = np.asarray(rl_vals)
    super_farmer_vals = np.asarray(super_farmer_vals)

    diffs_ceres_rl = np.empty(n_boot)
    diffs_rl_sf = np.empty(n_boot)

    for i in range(n_boot):
        samp_c = rng.choice(ceres_vals, size=len(ceres_vals), replace=True)
        samp_r = rng.choice(rl_vals, size=len(rl_vals), replace=True)
        samp_sf = rng.choice(super_farmer_vals, size=len(super_farmer_vals), replace=True)
        diffs_ceres_rl[i] = np.median(samp_c) - np.median(samp_r)
        diffs_rl_sf[i] = np.median(samp_r) - np.median(samp_sf)

    if metric == 'Reward':
        p_ceres_rl = np.sum(diffs_ceres_rl <= 0) / n_boot
        p_rl_sf = np.sum(diffs_rl_sf <= 0) / n_boot
    elif metric in ['Pesticide', 'Risk']:
        p_ceres_rl = np.sum(diffs_ceres_rl > 0) / n_boot
        p_rl_sf = np.sum(diffs_rl_sf > 0) / n_boot
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return {
        "Ceres,RL": {
            "median": np.median(diffs_ceres_rl),
            "ci_lower": np.percentile(diffs_ceres_rl, 2.5),
            "ci_upper": np.percentile(diffs_ceres_rl, 97.5),
            "p": p_ceres_rl,
        },
        "RL,SF": {
            "median": np.median(diffs_rl_sf),
            "ci_lower": np.percentile(diffs_rl_sf, 2.5),
            "ci_upper": np.percentile(diffs_rl_sf, 97.5),
            "p": p_rl_sf,
        },
    }


def tab3_collect_statistics(
        pkl_dir=None,
        pkl_dir_baselines=None,
        n_boot=10000,
        random_seed=42,
):
    results_dict, baselines_dict, random_dict = _tab3_load_pickles(
        pkl_dir=pkl_dir,
        pkl_dir_baselines=pkl_dir_baselines,
    )

    random_extracted = extract_metrics(random_dict)
    rl_extracted = extract_metrics(results_dict)
    baselines_extracted = extract_metrics(baselines_dict)

    baseline_agents = ["Ceres", "SPractice", "FPractice", "Zero"]
    metrics = ['Reward', 'Pesticide', 'Risk']
    agent_display_names = {
        "Ceres": "Ceres",
        "RL": "RL",
        "SPractice": "Super Farmer",
        "FPractice": "Farmer's Practice",
        "Random": "Random",
        "Zero": "Zero",
    }

    rng = np.random.default_rng(random_seed)
    agent_stats = {}
    delta_stats = {}

    for metric in metrics:
        baseline_by_year = {
            year: dict(zip(baseline_agents, vals))
            for year, vals in baselines_extracted[metric].items()
        }
        rl_by_year = rl_extracted[metric]

        data = {}
        for agent in baseline_agents:
            data[agent_display_names[agent]] = [
                baseline_by_year[year][agent]
                for year in sorted(baseline_by_year)
            ]

        data["RL"] = []
        for year in sorted(rl_by_year):
            data["RL"].extend(rl_by_year[year])

        data["Random"] = []
        for year in sorted(random_extracted[metric]):
            data["Random"].extend(random_extracted[metric][year])

        agent_stats[metric] = {
            agent: _tab3_bootstrap_median_ci(vals, n_boot=n_boot, rng=rng)
            for agent, vals in data.items()
        }

        ceres_vals = np.array([
            baseline_by_year[year]["Ceres"]
            for year in sorted(baseline_by_year)
        ])
        super_farmer_vals = np.array([
            baseline_by_year[year]["SPractice"]
            for year in sorted(baseline_by_year)
        ])
        rl_vals = np.concatenate([
            rl_by_year[year]
            for year in sorted(rl_by_year)
        ])
        delta_stats[metric] = _tab3_bootstrap_differences(
            ceres_vals,
            rl_vals,
            super_farmer_vals,
            metric,
            n_boot=n_boot,
            rng=rng,
        )

    return {
        "agents": agent_stats,
        "deltas": delta_stats,
    }


def _tab3_split_number(value, decimals):
    formatted = f"{value:.{decimals}f}"
    if float(formatted) == 0.0:
        formatted = formatted.replace("-", "")
    integer, fraction = formatted.split(".")
    return f"{integer} & {fraction}"


def _tab3_stat_columns(stat, decimals):
    return (
        f"{_tab3_split_number(stat['median'], decimals)} & "
        f"({_tab3_split_number(stat['ci_lower'], decimals)} & "
        f"{_tab3_split_number(stat['ci_upper'], decimals)})"
    )


def _tab3_p_value(p_value):
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def _tab3_agent_row(agent_name, stats, decimals_by_metric):
    metric_columns = [
        _tab3_stat_columns(stats[metric][agent_name], decimals_by_metric[metric])
        for metric in ['Reward', 'Pesticide', 'Risk']
    ]
    return f"{agent_name}\n  & " + "\n  & ".join(metric_columns) + r" \\"


def _tab3_delta_row(delta_label, delta_key, stats, decimals_by_metric):
    metric_columns = [
        _tab3_stat_columns(stats[metric][delta_key], decimals_by_metric[metric])
        for metric in ['Reward', 'Pesticide', 'Risk']
    ]
    return f"{delta_label}\n  & " + "\n  & ".join(metric_columns) + r" \\"


def _tab3_p_row(delta_key, stats):
    p_values = [
        _tab3_p_value(stats[metric][delta_key]["p"])
        for metric in ['Reward', 'Pesticide', 'Risk']
    ]
    return (
        f"  & \\multicolumn{{4}}{{l}}{{{p_values[0]}}} & \\multicolumn{{2}}{{l}}{{}}\n"
        f"  & \\multicolumn{{4}}{{l}}{{{p_values[1]}}} & \\multicolumn{{2}}{{l}}{{}}\n"
        f"  & \\multicolumn{{4}}{{l}}{{{p_values[2]}}} & \\multicolumn{{2}}{{l}}{{}} \\\\"
    )


def tab3_latex(
        pkl_dir=None,
        pkl_dir_baselines=None,
        n_boot=10000,
        random_seed=42,
        country_name="Spain",
        label="tab:results",
        print_table=True,
):
    stats = tab3_collect_statistics(
        pkl_dir=pkl_dir,
        pkl_dir_baselines=pkl_dir_baselines,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    decimals_by_metric = {
        "Reward": 2,
        "Pesticide": 1,
        "Risk": 3,
    }

    agent_rows = [
        _tab3_agent_row(agent_name, stats["agents"], decimals_by_metric)
        for agent_name in ["Ceres", "RL", "Super Farmer", "Farmer's Practice", "Random", "Zero"]
    ]
    delta_ceres_rl = _tab3_delta_row(r"$\Delta_{\text{Ceres,RL}}$", "Ceres,RL", stats["deltas"], decimals_by_metric)
    delta_rl_sf = _tab3_delta_row(r"$\Delta_{\text{RL,SF}}$", "RL,SF", stats["deltas"], decimals_by_metric)

    table = "\n".join([
        r"\begin{table}[!htbp]",
        (
            r"\caption{Statistics of reward, pesticide application and risk index aggregated over all test years, "
            f"showing median and 95\\% confidence intervals for {country_name}. "
            r"The highest possible reward is 0.0. Arrows indicate whether higher or lower values are desired.}"
        ),
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{",
        r"  l",
        r"  r@{.}l r@{.}l@{, }r@{.}l",
        r"  r@{.}l r@{.}l@{, }r@{.}l",
        r"  r@{.}l r@{.}l@{, }r@{.}l",
        r"}",
        r"\toprule",
        r"Agent",
        r"  & \multicolumn{6}{c}{Reward $(\uparrow)$}",
        r"  & \multicolumn{6}{c}{Pesticide Use $(\downarrow)$}",
        r"  & \multicolumn{6}{c}{Risk Index $(\downarrow)$} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-13} \cmidrule(lr){14-19}",
        r"  & \multicolumn{2}{c}{Median}",
        r"  & \multicolumn{4}{c}{95\% CI}",
        r"  & \multicolumn{2}{c}{Median}",
        r"  & \multicolumn{4}{c}{95\% CI}",
        r"  & \multicolumn{2}{c}{Median}",
        r"  & \multicolumn{4}{c}{95\% CI} \\",
        r"\midrule",
        *agent_rows,
        r"\midrule",
        delta_ceres_rl,
        _tab3_p_row("Ceres,RL", stats["deltas"]),
        r"\addlinespace",
        delta_rl_sf,
        _tab3_p_row("RL,SF", stats["deltas"]),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])

    if print_table:
        print(table)
    return table


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--trunc', action='store_true')
    arg = argparse.parse_args()
    main(arg)
