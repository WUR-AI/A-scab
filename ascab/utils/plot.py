import matplotlib.pyplot as plt


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

    # Iterate over each variable and create a subplot for it
    for i, variable in enumerate(variables):
        ax = axes[i] if num_variables > 1 else axes  # If only one variable, axes is not iterable
        ax.plot(results_df['Date'], results_df[variable], label=variable)
        ax.set_ylabel(f'{variable}')
        ax.legend()

        if variable == 'LeafWetness':
            ax.fill_between(results_df['Date'], results_df[variable], where=(results_df[variable] >= 0), color='blue',
                            alpha=0.3)

        # Add vertical line when the variable first passes the threshold
        thresholds = [0.016, 0.99]
        if variable == 'AscosporeMaturation' and thresholds is not None:
            for threshold in thresholds:
                exceeding_indices = results_df[results_df[variable] > threshold].index
                if len(exceeding_indices) > 0:
                    first_pass_index = exceeding_indices[0]
                    x_coordinate = results_df.loc[first_pass_index, 'Date']  # Get the corresponding date value
                    ax.axvline(x=x_coordinate, color='red', linestyle='--', label=f'Threshold ({threshold})')

    plt.xlabel('Date')
    plt.suptitle('Model Values Over Time')
    plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.0)  # Adjust vertical spacing between subplots
    plt.show()