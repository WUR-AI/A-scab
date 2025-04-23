import os
import pickle
from shutil import which

import numpy as np
import matplotlib.pyplot as plt

def separate_underscore(string, index = 0):
    string = string.split('_')
    return string[index]


def extract_metrics(results_dict):

    dict_extracted = {"Reward": {}, "Pesticide": {}}
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

        years_list.append(k)

    return dict_extracted


def main():

    this_file_path = os.path.abspath(__file__)
    pkl_dir = os.path.join(os.path.dirname(this_file_path), 'results')
    baseline_pickle_names = ['ceres.pkl', 'random.pkl', 'umbrella.pkl', 'zero.pkl']

    results_dict = {}
    baselines_dict = {}

    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl') and filename not in baseline_pickle_names:
            with open(os.path.join(pkl_dir, filename), 'rb') as f:
                # assumes filename is something like 'rl_agent_DQN_seed1_1.pkl'
                results_dict[separate_underscore(filename, -2)] = pickle.load(f)

        elif filename in baseline_pickle_names:
            with open(os.path.join(pkl_dir, filename), 'rb') as f:
                baselines_dict[filename[:-4]] = pickle.load(f)


    dict_extracted = extract_metrics(results_dict)
    baselines_extracted = extract_metrics(baselines_dict)

    baseline_names = ["Ceres", "Random", "Umbrella", "Zero"]


    for category in ["Reward", "Pesticide"]:
        print(f"\n{category} stats per year:")
        for year in sorted(dict_extracted[category].keys()):
            values = dict_extracted[category][year]
            mean_val = np.mean(values)
            std_val = np.std(values)
            # median_val = np.median(values)
            # iqr_val = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            print(f"Year {year}: RL mean = {mean_val:.3f}, std = {std_val:.4f}")
            for i, baseline_name in enumerate(baseline_names):
                value = baselines_extracted[category][year][i]
                print(f"           {baseline_name} = {value:.3f}")

    plot_normalized_ceres(dict_extracted, baselines_extracted)

if __name__ == '__main__':
    main()