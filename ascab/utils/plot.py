import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_results(results_df, variables=None):
    if variables is None:
        variables = results_df.columns.tolist()
    else:
        # Check if the provided variables exist in the DataFrame
        missing_variables = [var for var in variables if var not in results_df.columns]
        if missing_variables:
            raise ValueError(f"The following variables do not exist in the DataFrame: {', '.join(missing_variables)}")

    # Exclude 'Date' column from variables to be plotted
    variables = [var for var in variables if var != 'Date']
    variables.reverse()  # Reverse the order of the variables
    num_variables = len(variables)
    fig, axes = plt.subplots(num_variables, 1, figsize=(7, 4 * num_variables), sharex=True)

    # Find the closest date to April 1st in the dataset
    april_first = results_df['Date'] + pd.DateOffset(month=4, day=1)
    closest_april_first = april_first[april_first <= results_df['Date']].max()

    # Calculate the day number since April 1st
    results_df['DayNumber'] = (results_df['Date'] - closest_april_first).dt.days


    # Iterate over each variable and create a subplot for it
    for i, variable in enumerate(variables):
        ax = axes[i] if num_variables > 1 else axes  # If only one variable, axes is not iterable
        ax.plot(results_df['Date'], results_df[variable], label=variable)
        ax.set_ylabel(f'{variable}')
        ax.legend()

        if variable == 'LeafWetness':
            ax.fill_between(results_df['Date'], results_df[variable], where=(results_df[variable] >= 0), color='blue',
                            alpha=0.3)

        if variable == 'Precipitation':
            ax.axhline(y=0.2, color='red', linestyle='--')

        # Add vertical line when the variable first passes the threshold
        thresholds = [0.016, 0.99]
        if variable == 'AscosporeMaturation' and thresholds is not None:
            for threshold in thresholds:
                exceeding_indices = results_df[results_df[variable] > threshold].index
                if len(exceeding_indices) > 0:
                    first_pass_index = exceeding_indices[0]
                    x_coordinate = results_df.loc[first_pass_index, 'Date']  # Get the corresponding date value
                    ax.axvline(x=x_coordinate, color='red', linestyle='--', label=f'Threshold ({threshold})')

        if i == num_variables - 1:  # Only add secondary x-axis to the bottom subplot
            # Add secondary x-axis with limited ticks starting from day 0
            tick_interval = 25
            secax = ax.secondary_xaxis('bottom', color='grey')

            # Determine the closest date to April 1st to start the ticks
            start_date = pd.Timestamp('2011-04-01')
            start_index = results_df.index[results_df['Date'] >= start_date][0]

            # Generate tick locations and labels based on the start_index and tick_interval
            tick_locations = results_df['Date'].iloc[start_index::tick_interval]
            tick_labels = results_df['DayNumber'].iloc[start_index::tick_interval]

            secax.set_xticks(tick_locations)
            secax.set_xticklabels(tick_labels)
            # Adjust tick label rotation and alignment
            secax.tick_params(axis='x', labelrotation=0, direction='in')


    plt.xlabel('Date')
    plt.suptitle('Model Values Over Time')
    plt.xticks(rotation=45)


    plt.subplots_adjust(hspace=0.0)  # Adjust vertical spacing between subplots
    plt.show()


def plot_precipitation_with_rain_event(df_hourly, day):
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
