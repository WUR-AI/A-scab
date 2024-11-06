import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Union
from ascab.model.infection import InfectionRate, get_pat_threshold


def plot_results(results: [Union[dict[str, pd.DataFrame], pd.DataFrame]], variables: list[str] = None, save_path: str = None):
    results = {"": results} if not isinstance(results, dict) else results
    alpha = 1.0 if len(results) == 1 else 0.5

    if variables is None:
        variables = list(results.values())[0].columns.tolist()
        variables.reverse()  # Reverse the order of the variables
    else:
        # Check if the provided variables exist in the DataFrames
        for df in results.values():
            missing_variables = [var for var in variables if var not in df.columns]
            if missing_variables:
                raise ValueError(
                    f"The following variables do not exist in the DataFrame: {', '.join(missing_variables)}"
                )

    # Exclude 'Date' column from variables to be plotted
    variables = [var for var in variables if var != 'Date']

    num_variables = len(variables)
    fig, axes = plt.subplots(num_variables, 1, figsize=(7, 3 * num_variables), sharex=True)
    for index_results, (df_key, df) in enumerate(results.items()):
        if "Reward" in df.columns:
            reward = df["Reward"].sum()
            reward_string = (f"Reward: {reward.item():.2f}" if isinstance(reward, np.ndarray) and reward.size == 1 else f"{reward:.2f}")
        else:
            reward_string = "N/A"
        # Iterate over each variable and create a subplot for it
        for i, variable in enumerate(variables):
            ax = axes[i] if num_variables > 1 else axes  # If only one variable, axes is not iterable

            if index_results == 0:
                ax.text(0.015, 0.85, variable, transform=ax.transAxes, verticalalignment="top",horizontalalignment="left",
                        bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.25'))

            ax.step(df['Date'], df[variable], label=f'{df_key} {reward_string}', where='post', alpha=alpha)
            if i == 0: ax.legend()

            if variable == 'LeafWetness':
                ax.axhline(y=8.0, color="red", linestyle="--")
            if variable == 'Precipitation':
                ax.axhline(y=0.2, color='red', linestyle='--')
            if variable == 'TotalRain':
                ax.axhline(y=0.25, color='red', linestyle='--')
            if variable == 'HumidDuration':
                ax.axhline(y=8.0, color="red", linestyle="--")

            # Add vertical line when the variable first passes the threshold
            thresholds = [get_pat_threshold(), 0.99]
            if variable == 'AscosporeMaturation' and thresholds is not None:
                for threshold in thresholds:
                    exceeding_indices = df[df[variable] > threshold].index
                    if len(exceeding_indices) > 0:
                        first_pass_index = exceeding_indices[0]
                        x_coordinate = df.loc[first_pass_index, 'Date']  # Get the corresponding date value
                        ax.axvline(x=x_coordinate, color='red', linestyle='--', label=f'Threshold ({threshold})')


    ax = axes[-1] if num_variables > 1 else axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=0)
    plt.setp(ax.get_xticklabels(), ha="center")
    all_years = pd.to_datetime(
        pd.concat([df["Date"] for df in results.values()])
    ).dt.year.unique()
    plt.xlabel(f'{all_years}')
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_infection(infection: InfectionRate):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Create figure and axis for the first plot

    ax1.plot(infection.hours, infection.s1_sigmoid, linestyle='dotted', label='sigmoid1', color='blue')
    ax1.plot(infection.hours, infection.s2_sigmoid, linestyle='dotted', label='sigmoid2', color='purple')
    ax1.plot(infection.hours, infection.s3_sigmoid, linestyle='dotted', label='sigmoid3', color='green')

    ax1.plot(infection.hours, infection.s1, label='s1', linestyle='solid', color='blue')
    ax1.plot(infection.hours, infection.s2, label='s2', linestyle='solid', color='purple')
    ax1.plot(infection.hours, infection.s3, label='s3', linestyle='solid', color='green')
    ax1.plot(infection.hours, infection.total_population, label='population', linestyle='solid', color='yellow')

    total = np.sum([infection.s1, infection.s2, infection.s3], axis=0)
    ax1.plot(infection.hours, total, label='sum_s1_s2_s3', linestyle='solid', color='black')
    ax1.axvline(x=0, color="red", linestyle="--")
    ax1.axvline(x=infection.infection_duration, color="red", linestyle="--", label="infection duration")

    discharge_duration = 90.96 * infection.infection_temperature **(-0.96)
    ax1.axvline(x=discharge_duration, color="orange", linestyle="--", label="discharge duration")

    if len(infection.hours) % 24 == 0:
        ax1.step([item for sublist in [infection.hours[index*24: min(len(infection.hours), (index+1)*24)] for index, _ in enumerate(infection.risk)] for item in sublist],
             [item for sublist in [[entry[1]] * 24 for entry in infection.risk] for item in sublist],
             color="orange", linestyle='solid', label="cumulative risk", where='post')

    dates = infection.discharge_date + pd.to_timedelta(infection.hours, unit="h")
    unique_dates = pd.date_range(start=dates[0], end=dates[-1], freq="D")

    if len(infection.hours) % 24 == 0:
        for i, unique_date in enumerate(unique_dates):
            ax1.axvline(x=infection.hours[i*24], color="grey", linestyle="--", linewidth=0.8)
            ax1.text(infection.hours[i*24]+0.1, ax1.get_ylim()[1], unique_date.strftime("%Y-%m-%d"),
                 color="grey", ha="left", va="top", rotation=90, fontsize=9)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Infection Data {infection.risk[-1][1]:.2f} {infection.infection_temperature:.1f} {infection.infection_duration}')
    plt.legend()
    plt.show()


def plot_precipitation_with_rain_event(df_hourly: pd.DataFrame, day: pd.Timestamp):
    # Filter the DataFrame for the specific day
    df_day = df_hourly[df_hourly['Hourly Date'].dt.date == day.date()]
    # Plot the precipitation
    plt.figure(figsize=(7, 4))
    # datetime_objects = [datetime.fromisoformat(dt[:-6]) for dt in datetime_values]

    plt.step(df_day['Hourly Date'], df_day['Hourly Precipitation'], where='post', label='Hourly Precipitation')

    # Plot filled area for rain event
    for idx, row in df_day.iterrows():
        if row['Hourly Rain Event']:
            plt.axvspan(row['Hourly Date'], row['Hourly Date'] + pd.Timedelta(hours=1), color='gray', alpha=0.3)

    plt.xlabel('Hour of the Day')
    plt.ylabel('Precipitation')
    plt.title(f'Precipitation and Rain Event for {day.date()}')
    plt.legend()
    plt.grid(True)

    # Set minor ticks to represent hours
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    # Enable minor grid lines
    plt.grid(which='minor', linestyle='--', linewidth=0.5)

    # Format major ticks to hide day information
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # Set y-range to start from 0
    plt.ylim(0, max(0.21, max(df_day['Hourly Precipitation']) + 0.1))

    # Add horizontal line at y = 0.2
    plt.axhline(y=0.2, color='red', linestyle='--', label='Threshold')
    plt.show()
