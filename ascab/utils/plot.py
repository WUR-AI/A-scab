import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import pandas as pd
import numpy as np
from typing import Union
from ascab.model.infection import InfectionRate, get_pat_threshold


def get_default_plot_variables() -> list:
    return ["Precipitation", "AscosporeMaturation", "Discharge", "Pesticide", "Risk", "Action"]


def plot_results(results: [Union[dict[str, pd.DataFrame], pd.DataFrame]],
                 variables: list[str] = get_default_plot_variables(),
                 save_path: str = None,
                 fig_size: int = 10,
                 save_type: str = 'png',
                 per_year: bool = True,
                 zoom: bool = True,
                 stacked: bool = True,):
    results = {"": results} if not isinstance(results, dict) else results
    alpha = 1.0 if len(results) == 1 else 0.5

    if variables is None:
        variables = list(results.values())[0].columns.tolist()
        variables.reverse()
    else:
        for df in results.values():
            missing_variables = [var for var in variables if var not in df.columns]
            if missing_variables:
                raise ValueError(
                    f"The following variables do not exist in the DataFrame: {', '.join(missing_variables)}"
                )

    # Exclude 'Date' column from variables to be plotted
    variables = [var for var in variables if var != 'Date']
    num_variables = len(variables)

    if per_year is False:
        fig, axes = plt.subplots(num_variables, 1, figsize=(fig_size, num_variables), sharex=True)

        if len(results.keys()) > 1 and not per_year:
            for index_results, (df_key, df) in enumerate(results.items()):
                if "Reward" in df.columns:
                    df['Year'] = df['Date'].dt.year
                    reward_per_year = df.groupby('Year')['Reward'].sum()
                    reward_string = " | ".join([f"{year}: {total:.2f}" for year, total in reward_per_year.items()])
                else:
                    reward_string = "N/A"
                # Iterate over each variable and create a subplot for it
                for i, variable in enumerate(variables):
                    ax = axes[i] if num_variables > 1 else axes  # If only one variable, axes is not iterable

                    if index_results == 0:
                        ax.text(0.015, 0.85, variable, transform=ax.transAxes, verticalalignment="top",horizontalalignment="left",
                                bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.25'))
                    df['Date'] = df['Date'].apply(lambda d: d.replace(year=2000))  # put all years on top of each other
                    # Find where the date resets (i.e., next date is earlier than the current one)
                    date_resets = df['Date'].diff().dt.total_seconds() < 0
                    reset_indices = date_resets[date_resets].index - 1
                    df.loc[reset_indices, variable] = np.nan
                    ax.step(df['Date'], df[variable], label=f'{df_key} {reward_string}', where='post', alpha=alpha)
                    if i == (len(variables) - 1):
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

                    if variable == 'LeafWetness':
                        ax.axhline(y=8.0, color="red", linestyle="--")
                    if variable == 'Precipitation':
                        ax.axhline(y=0.2, color='red', linestyle='--')
                    if variable == 'TotalRain':
                        ax.axhline(y=0.25, color='red', linestyle='--')
                    if variable == 'HumidDuration':
                        ax.axhline(y=8.0, color="red", linestyle="--")

        else:
            # We know there's exactly one DataFrame in results
            df_key, df = next(iter(results.items()))

            # extract years & sum rewards by year
            if "Reward" in df.columns:
                df['Year'] = df['Date'].dt.year
                reward_per_year = df.groupby('Year')['Reward'].sum().to_dict()
            else:
                reward_per_year = {}

            # pick a colormap
            cmap = plt.get_cmap('tab10')
            years = sorted(df['Year'].unique())

            # for each year, plot its data in a different color
            for idx, year in enumerate(years):
                color = cmap(idx % cmap.N)
                df_year = df.loc[df['Year'] == year, :].copy()
                # align all years to the same 2000-base for step-plot
                df_year['Date'] = df_year['Date'].apply(lambda d: d.replace(year=2000))

                for i, variable in enumerate(variables):
                    ax = axes[i] if num_variables > 1 else axes

                    ax.text(0.015, 0.85, variable,
                            transform=ax.transAxes,
                            verticalalignment="top",
                            horizontalalignment="left",
                            bbox=dict(facecolor='white',
                                      edgecolor='lightgrey',
                                      boxstyle='round,pad=0.25'))

                    ax.step(
                        df_year['Date'],
                        df_year[variable],
                        where='post',
                        color=color,
                        label=f"{year}: {reward_per_year.get(year, 0):.2f}",
                        alpha=alpha
                    )

                    # draw thresholds & maturation-lines as before
                    if variable == 'LeafWetness':
                        ax.axhline(y=8.0, color="red", linestyle="--")
                    elif variable == 'Precipitation':
                        ax.axhline(y=0.2, color='red', linestyle='--')
                    elif variable == 'TotalRain':
                        ax.axhline(y=0.25, color='red', linestyle='--')
                    elif variable == 'HumidDuration':
                        ax.axhline(y=8.0, color="red", linestyle="--")

                    if variable == 'AscosporeMaturation':
                        for threshold in [get_pat_threshold(), 0.99]:
                            exceed = df_year[df_year[variable] > threshold]
                            if not exceed.empty:
                                x0 = exceed.iloc[0]['Date']
                                ax.axvline(x=x0, color='red', linestyle='--')

                # only add one legend per subplot
                if num_variables > 1:
                    legend_ax = axes[-1]
                else:
                    legend_ax = axes
                legend_ax.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=min(len(years), 4),
                    frameon=False
                )
                legend_ax.text(
                    0.5,
                    -0.7,
                    df_key,  # your lone-dict key
                    transform=legend_ax.transAxes,
                    ha='center',  # horizontal center
                    va='top',
                    fontsize='medium',
                    fontweight='light'
                )

        ax = axes[-1] if num_variables > 1 else axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate(rotation=0)
        plt.setp(ax.get_xticklabels(), ha="center")

        if save_path:
            print(f'save {save_path}')
            plt.savefig(save_path, format=save_type, dpi=200, bbox_inches='tight')
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    else:  # if year is True
        if not zoom and not stacked:
            cmap = plt.get_cmap('tab10')

            # assume all your dfs have a 'Year' column already; if not, add it:
            for df in results.values():
                if "Year" not in df.columns:
                    df["Year"] = df["Date"].dt.year

            # figure out which years appear anywhere
            all_years = sorted(
                set().union(*(df["Year"].unique() for df in results.values()))
            )

            for year in all_years:
                # 1) prepare a figure with one subplot per variable
                fig, axes = plt.subplots(
                    num_variables, 1,
                    figsize=(fig_size, num_variables),
                    sharex=True
                )

                # 2) for each key, filter & save that year's data, then plot it
                for idx, (df_key, df) in enumerate(results.items()):
                    color = cmap(idx % cmap.N)
                    df_year = df[df["Year"] == year].copy()
                    if df_year.empty:
                        continue

                    # (a) save the raw year‐filtered data to CSV
                    # csv_path = os.path.join(out_dir, f"{df_key}_{year}.csv")
                    # df_year.to_csv(csv_path, index=False)

                    # (b) if you want reward‐sums per year in the legend:
                    if "Reward" in df_year.columns:
                        total_reward = df_year["Reward"].sum()
                        legend_label = f"{df_key}: {total_reward:.2f}"
                    else:
                        legend_label = df_key

                    # (c) normalize dates to 2000 so years overlap
                    # df_year["DatePlot"] = df_year["Date"].apply(lambda ts: ts.replace(year=2000))

                    # (d) plot each variable for this key/year
                    for i, variable in enumerate(variables):
                        ax = axes[i] if num_variables > 1 else axes
                        ax.step(
                            df_year["Date"],
                            df_year[variable],
                            where="post",
                            label=legend_label,
                            alpha=alpha,
                            color=color,
                        )
                        # redraw your thresholds & maturation‐lines:
                        if variable == "LeafWetness":
                            ax.axhline(8.0, linestyle="--", color="red")
                        elif variable == "Precipitation":
                            ax.axhline(0.2, linestyle="--", color="red")
                        elif variable == "TotalRain":
                            ax.axhline(0.25, linestyle="--", color="red")
                        elif variable == "HumidDuration":
                            ax.axhline(8.0, linestyle="--", color="red")

                        if variable == "AscosporeMaturation":
                            for thresh in [get_pat_threshold(), 0.99]:
                                exceed = df_year[df_year[variable] > thresh]
                                if not exceed.empty:
                                    x0 = exceed.iloc[0]["Date"]
                                    ax.axvline(x0, linestyle="--", color="red")

                    # if zoom is False:
                    # 3) finish each subplot
                for i, variable in enumerate(variables):
                    ax = axes[i] if num_variables > 1 else axes
                    # add the variable name in the first column
                    ax.text(
                        0.015, 0.85, variable,
                        transform=ax.transAxes,
                        va="top", ha="left",
                        bbox=dict(facecolor="white",
                                  edgecolor="lightgrey",
                                  boxstyle="round,pad=0.25")
                    )
                # unified legend on the bottom subplot
                legend_ax = axes[-1] if num_variables > 1 else axes
                legend_ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=len(results),
                    frameon=False
                )
                if save_path:
                    out_path = os.path.join(os.path.dirname(save_path), f"plot_{year}.png")
                    print(f'save {out_path}')
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=200)
                plt.show()
                plt.close(fig)
        elif zoom and stacked:
            cmap = plt.get_cmap('tab10')

            # assume all your dfs have a 'Year' column already; if not, add it:
            for df in results.values():
                if "Year" not in df.columns:
                    df["Year"] = df["Date"].dt.year

            # figure out which years appear anywhere
            all_years = sorted(
                set().union(*(df["Year"].unique() for df in results.values()))
            )

            for year in all_years:
                start_date, end_date = get_thresholds_per_year(year, results)
                # 0) Make "master figure"
                fig_combined = plt.figure(constrained_layout=True, figsize=(fig_size+2, num_variables),)

                subfig_left, subfig_right = fig_combined.subfigures(1, 2, width_ratios=[2, 1])

                # 1) prepare a figure with one subplot per variable

                gs = subfig_left.add_gridspec(
                    nrows=num_variables, ncols=1,
                    height_ratios=[1 for _ in range(num_variables)],
                    hspace=.22, wspace=.22
                )

                axes_left = [
                    subfig_left.add_subplot(gs[i, :]) for i, _ in enumerate(variables)
                ]

                # 2) for each key, filter & save that year's data, then plot it
                for idx, (df_key, df) in enumerate(results.items()):
                    color = cmap(idx % cmap.N)
                    df_year = df[df["Year"] == year].copy()
                    if df_year.empty:
                        continue

                    # (b) if you want reward‐sums per year in the legend:
                    if "Reward" in df_year.columns:
                        total_reward = df_year["Reward"].sum()
                        legend_label = f"{df_key}: {total_reward:.2f}"
                    else:
                        legend_label = df_key

                    # (c) normalize dates to 2000 so years overlap
                    # df_year["DatePlot"] = df_year["Date"].apply(lambda ts: ts.replace(year=2000))

                    # (d) plot each variable for this key/year
                    risk_date = []
                    for i, variable in enumerate(variables):
                        ax = axes_left[i]
                        ax.step(
                            df_year["Date"],
                            df_year[variable],
                            where="post",
                            label=legend_label,
                            alpha=alpha,
                            color=color,
                        )
                        # redraw your thresholds & maturation‐lines:
                        if variable == "LeafWetness":
                            ax.axhline(8.0, linestyle="--", color="red")
                        elif variable == "Precipitation":
                            ax.axhline(0.2, linestyle="--", color="red")
                        elif variable == "TotalRain":
                            ax.axhline(0.25, linestyle="--", color="red")
                        elif variable == "HumidDuration":
                            ax.axhline(8.0, linestyle="--", color="red")

                        if variable in ["AscosporeMaturation"]:
                            for thresh in [get_pat_threshold(), 0.99]:
                                exceed = df_year[df_year[variable] > thresh]
                                if not exceed.empty:
                                    risk_date.append(exceed.iloc[0]["Date"])
                                    ax.axvline(risk_date[1] if thresh == 0.99 else risk_date[0], linestyle="--", color="red")
                        elif variable in ["Pesticide", "Risk", "Action"]:
                                ax.axvline(start_date, linestyle="--", color="red")
                                ax.axvline(end_date, linestyle="--", color="red")

                    # 3) finish each subplot
                    for i, variable in enumerate(variables):
                        ax = axes_left[i]
                        # add the variable name in the first column
                        ax.text(
                            0.015, 0.85, variable,
                            transform=ax.transAxes,
                            va="top", ha="left",
                            bbox=dict(facecolor="white",
                                      edgecolor="lightgrey",
                                      boxstyle="round,pad=0.25")
                        )
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

                # ------------------- stacked part

                _, axes_zoom = make_year_plot(year, results, fig_size=fig_size, alpha=alpha,
                                                     stacked=stacked, container=subfig_right)

                for i, variable in enumerate(variables):
                    if variable in ['Pesticide', 'Risk', 'Action']:
                        ax = axes_zoom[variable]
                        # add the variable name in the first column
                        ax.text(
                            0.05, 0.95, variable,
                            transform=ax.transAxes,
                            va="top", ha="left",
                            bbox=dict(facecolor="white",
                                      edgecolor="lightgrey",
                                      boxstyle="round,pad=0.25")
                        )

                handles, labels = [], []
                for ax in fig_combined.axes:  # fig_combined is the outer Figure
                    h, l = ax.get_legend_handles_labels()
                    handles.extend(h)
                    labels.extend(l)

                # keep only the first occurrence of each label
                by_label = dict(zip(labels, handles))

                # ── create one combined legend at the bottom centre ────────────────────────
                fig_combined.legend(
                    by_label.values(), by_label.keys(),
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.05),  # y < 0 ⇒ place *below* the figure
                    ncol=min(5, len(by_label)),  # wrap into rows if many entries
                    frameon=False,
                    bbox_transform=fig_combined.transFigure
                )

                for i, variable in enumerate(variables):
                    if variable in ['Pesticide', 'Risk', 'Action']:
                        # diagonal from bottom-left to top-right of the *whole* canvas
                        color_line = 'red'

                        for edge in [0, 1]:
                            # Get left coords
                            x0_num = mdates.date2num(end_date if edge == 0 else start_date)
                            y0 = axes_left[i].get_ylim()[1]

                            pt0 = (x0_num, y0)

                            # rigt coords
                            pt1 = (0, edge)

                            # Connect with this package
                            conn = ConnectionPatch(
                                xyA=pt0, coordsA=axes_left[i].transData,  # left axis (data coords)
                                xyB=pt1, coordsB=axes_zoom[variable].transAxes,  # right axis (axes coords)
                                axesA=axes_left[i], axesB=axes_zoom[variable],
                                color=color_line, lw=1, ls="--"
                            )
                            fig_combined.add_artist(conn)

                if save_path:
                    out_path = os.path.join(os.path.dirname(save_path), f"plot_{year}_stacked.png")
                    print(f'save {out_path}')
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=200)
                plt.show()
                plt.close(fig_combined)

        else:  # if zoom is True:
            for year in sorted(set().union(*(df["Year"].unique()
                                           for df in results.values()))):
                fig, axes = make_year_plot(year, results, fig_size=10, alpha=alpha)
                if fig is None:
                    continue

                for i, variable in enumerate(variables):
                    ax = list(axes.values())[i]
                    # add the variable name in the first column
                    if variable in ['Pesticide', 'Risk', 'Action']:
                        ax.text(
                            0.05, 0.95, variable,
                            transform=ax.transAxes,
                            va="top", ha="left",
                            bbox=dict(facecolor="white",
                                      edgecolor="lightgrey",
                                      boxstyle="round,pad=0.25")
                        )
                    else:
                        ax.text(
                            0.015, 0.85, variable,
                            transform=ax.transAxes,
                            va="top", ha="left",
                            bbox=dict(facecolor="white",
                                      edgecolor="lightgrey",
                                      boxstyle="round,pad=0.25")
                        )
                # unified legend on the bottom subplot
                legend_ax = list(axes.values())[-2]
                legend_ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=len(results),
                    frameon=False,
                    fontsize="large"
                )

                if save_path:
                    out_path = os.path.join(save_path, f"plot_zoom_{year}.png")
                    print(f'save {out_path}')
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=200)
                plt.show()
                plt.close(fig)


def get_thresholds_per_year(year, results_dict):
    PAT_THR = get_pat_threshold()
    END_THR = 0.99  # full maturation

    start_date, end_date = None, None
    for df in results_dict.values():
        asc = df.loc[df["Year"] == year, "AscosporeMaturation"]
        # skip empty dfs (algorithm did not run that year)
        if asc.empty:
            continue
        s = df.loc[asc.gt(PAT_THR).idxmax(), "Date"]
        e = df.loc[asc.gt(END_THR).idxmax(), "Date"]
        start_date = s if start_date is None else min(start_date, s)
        end_date = e if end_date is None else max(end_date, e)

    if start_date is None or end_date is None:
        print(f"No data for {year}")
        return

    return start_date, end_date


def make_year_plot(year, results_dict, fig_size=9, alpha=0.5, stacked=False, container=None):
    """
    results_dict:  {name -> full-year dataframe}
                   each df must have columns
                   [Date, Precipitation, AscosporeMaturation, Discharge,
                    Pesticide, Risk, Action]
    """
    cmap = plt.get_cmap("tab10")  # one colour per algorithm
    # ── decide zoom window from the earliest start & latest end ──────────
    start_date, end_date = get_thresholds_per_year(year, results_dict)

    # ── layout 4×3  (top three rows span, bottom split) ──────────────────


    if stacked:
        if container is None:
            container = plt.figure(figsize=(fig_size, fig_size))

        gs = container.add_gridspec(
            nrows=3, ncols=1,
            height_ratios=[1, 1, 1],
            hspace=.22, wspace=.22
        )

        axes = {
            # "Precipitation": fig.add_subplot(gs[0, :]),
            # "AscosporeMaturation": fig.add_subplot(gs[1, :]),
            # "Discharge": fig.add_subplot(gs[2, :]),
            "Pesticide": container.add_subplot(gs[0, 0]),
            "Risk": container.add_subplot(gs[1, 0]),
            "Action": container.add_subplot(gs[2, 0]),
        }

        for idx, (name, df_full) in enumerate(results_dict.items()):
            df = df_full[df_full["Year"] == year]
            if df.empty:
                continue
            colour = cmap(idx % cmap.N)

            total_reward = df["Reward"].sum()

            for var, ax in axes.items():
                ax.step(df["Date"], df[var], where="post",
                        color=colour, alpha=alpha)

        # ── cosmetics / zoom bottom row, red lines, labels  ──────────────────
        for var, ax in axes.items():
            ax.set_ylabel(var)

            if var in ["Action", "Risk", "Pesticide"]:
                ax.set_xlim(start_date, end_date)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            else:
                ax.set_xlabel("")  # hide shared x-labels on upper rows
                ax.set_xticklabels([])

            ax.set_ylabel("")

        # fig.tight_layout()
        return container, axes


    else:
        fig = plt.figure(figsize=(fig_size, fig_size - 2))
        gs  = gridspec.GridSpec(4, 3, height_ratios=[.5,.5,.5,1.0],
                                hspace=.22, wspace=.22)

        axes = {
            "Precipitation"       : fig.add_subplot(gs[0, :]),
            "AscosporeMaturation" : fig.add_subplot(gs[1, :]),
            "Discharge"           : fig.add_subplot(gs[2, :]),
            "Pesticide"           : fig.add_subplot(gs[3, 0]),
            "Risk"                : fig.add_subplot(gs[3, 1]),
            "Action"              : fig.add_subplot(gs[3, 2]),
        }

        # ── plot every algorithm on the same axes ────────────────────────────
        for idx, (name, df_full) in enumerate(results_dict.items()):
            df = df_full[df_full["Year"] == year]
            if df.empty:
                continue
            colour = cmap(idx % cmap.N)

            total_reward = df["Reward"].sum()
            legend_label = f"{name}: {total_reward:.2f}"

            for var, ax in axes.items():
                ax.step(df["Date"], df[var], where="post",
                        label=legend_label if var=="Risk" else None,  # legend once
                        color=colour, alpha=alpha)

        # ── cosmetics / zoom bottom row, red lines, labels  ──────────────────
        for var, ax in axes.items():
            ax.set_ylabel(var)

            if var == "Precipitation":
                ax.axhline(0.2, linestyle="--", color="red")

            if var == "AscosporeMaturation":
                ax.axvline(start_date, color="red", ls="--")
                ax.axvline(end_date,   color="red", ls="--")

            if var in ["Pesticide","Risk","Action"]:
                ax.set_xlim(start_date, end_date)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                # for t in ax.get_xticklabels():
                #     t.set_rotation(45)
            elif var in ["Discharge"]:
                ...
            else:
                ax.set_xlabel("")          # hide shared x-labels on upper rows
                ax.set_xticklabels([])

            ax.set_ylabel("")

        # unified legend → use one of the bottom axes
        axes["Risk"].legend(loc="lower center",
                              bbox_to_anchor=(1.5, -0.28),   # centre under grid
                              ncol=min(5, len(results_dict)),
                              frameon=False)
        # fig.tight_layout()
        return fig, axes


def plot_infection(infection: InfectionRate):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Create figure and axis for the first plot

    ax1.plot(infection.hours, infection.s1_sigmoid, linestyle='dotted', label='sigmoid1', color='blue')
    ax1.plot(infection.hours, infection.s2_sigmoid, linestyle='dotted', label='sigmoid2', color='purple')
    ax1.plot(infection.hours, infection.s3_sigmoid, linestyle='dotted', label='sigmoid3', color='green')

    ax1.plot(infection.hours, infection.s1, label='s1', linestyle='solid', color='blue')
    ax1.plot(infection.hours, infection.s2, label='s2', linestyle='solid', color='purple')
    ax1.plot(infection.hours, infection.s3, label='s3', linestyle='solid', color='green')
    ax1.plot(infection.hours, infection.total_population, label='population', linestyle='solid', color='yellow')
    ax1.plot(infection.hours, infection.pesticide_levels, label='pesticide', linestyle='solid', color='brown')

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

def plot_normalized_reward(dict_extracted, baselines_extracted, random_extracted, plot_type='bar'):
    years = sorted(dict_extracted['Reward'].keys())
    cmap = plt.get_cmap('tab10')

    baseline_u = []  # Umbrella
    baseline_z = []  # Zero
    baseline_c = []
    random_distributions = []
    rl_distributions = []

    for yr in years:
        rl_raw = np.array(dict_extracted['Reward'][yr])
        random_raw = np.array(random_extracted['Reward'][yr])


        # baselines_extracted['Reward'][yr] == [ceres, umbrella, zero]
        ceres, umb, zro = baselines_extracted['Reward'][yr]
        lowest_rand = min(random_raw)
        worst = min(zro, lowest_rand, umb)
        baseline_c.append((ceres - worst) / (ceres - worst))  # =1
        baseline_u.append((umb - worst) / (ceres - worst))
        baseline_z.append((zro - worst) / (ceres - worst))

        # normalize RL seeds
        rl_norm = (rl_raw - worst) / (ceres - worst)
        rl_distributions.append(rl_norm)

        random_norm = (random_raw - worst) / (ceres - worst)
        random_distributions.append(random_norm)

    # # Compute medians and IQRs for RL
    # medians = [np.median(arr) for arr in rl_distributions]
    # q1s = [np.quantile(arr, 0.25) for arr in rl_distributions]
    # q3s = [np.quantile(arr, 0.75) for arr in rl_distributions]

    means = [np.mean(arr) for arr in rl_distributions]
    stds = [np.std(arr) for arr in rl_distributions]

    means_random = [np.mean(arr) for arr in random_distributions]
    stds_random = [np.std(arr) for arr in random_distributions]

    x = np.arange(len(years))
    offsets = {'Ceres': -0.2, 'RL': -0.1, 'Umbrella': 0.0, 'Random': 0.1, 'Zero': 0.2}

    fig, ax = plt.subplots()

    if plot_type == 'scatter':

        # scatter Ceres Umbrella & Zero
        size = 40
        ax.scatter(x + offsets['Ceres'], baseline_c, marker='o', s=size, label='Ceres', color='#e41a1c')
        # ax.scatter(x + offsets['Random'], baseline_r, marker='o', s=size, label='Random', color='#4daf4a')
        ax.errorbar(
            x + offsets['Random'],
            # medians,
            means_random,
            yerr=stds_random,
            fmt='o',
            capsize=5,
            label='Random (mean ± std)',
            color='#4daf4a',
        )
        ax.scatter(x + offsets['Umbrella'], baseline_u, marker='o', s=size, label='Umbrella', color='#ff7f00')
        ax.scatter(x + offsets['Zero'], baseline_z, marker='o', s=size, label='Zero', color='#a65628')

        # 2) RL medians + IQR errorbars
        # yerr_lower = [med - q1 for med, q1 in zip(medians, q1s)]
        # yerr_upper = [q3 - med for med, q3 in zip(medians, q3s)]
        # yerr = np.vstack([yerr_lower, yerr_upper])

        yerr = stds

        ax.errorbar(
            x + offsets['RL'],
            # medians,
            means,
            yerr=yerr,
            fmt='o',
            capsize=5,
            label='RL (mean ± std)',
            color='#377eb8',
        )
    elif plot_type == 'bar':
        alpha = 0.9
        # Define bar width and offsets
        width = 0.15
        offsets = {
            'Ceres': -2 * width,
            'Umbrella': -1 * width,
            'Zero': 0,
            'RL': 1 * width,
            'Random': 2 * width,
        }
        # Bars for baselines
        ax.bar(x + offsets['Ceres'], baseline_c, width, label='Ceres', color=cmap(2), alpha=alpha) #e41a1c
        ax.bar(x + offsets['Umbrella'], baseline_u, width, label='Umbrella', color=cmap(1), alpha=alpha) #ff7f00
        ax.bar(x + offsets['Zero'], baseline_z, width, label='Zero', color=cmap(0), alpha=alpha) #a65628

        # Bars with errorbars for distributions
        ax.bar(
            x + offsets['RL'],
            means,
            width,
            yerr=stds,
            capsize=5,
            label='RL (mean ± std)',
            color=cmap(3), #377eb8
            alpha=alpha
        )
        ax.bar(
            x + offsets['Random'],
            means_random,
            width,
            yerr=stds_random,
            capsize=5,
            label='Random (mean ± std)',
            color=cmap(4), #4daf4a
            alpha=alpha
        )

    else:
        raise ValueError("plot_type must be either 'scatter' or 'bar'")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1,11))
    ax.set_ylabel('Normalized Reward')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_pesticide_use(dict_extracted, baselines_extracted, random_extracted):
    years = sorted(dict_extracted['Pesticide'].keys())
    cmap = plt.get_cmap('tab10')
    alpha = 0.9

    baseline_u = []  # Umbrella
    baseline_z = []  # Zero
    baseline_c = []
    random_distributions = []
    rl_distributions = []

    for yr in years:
        rl_raw = np.array(dict_extracted['Pesticide'][yr])
        random_raw = np.array(random_extracted['Pesticide'][yr])


        # baselines_extracted['Reward'][yr] == [ceres, umbrella, zero]
        ceres, umb, zro = baselines_extracted['Pesticide'][yr]
        baseline_c.append(ceres)
        baseline_u.append(umb)
        baseline_z.append(zro)

        random_distributions.append(random_raw)
        rl_distributions.append(rl_raw)

    means = [np.mean(arr) for arr in rl_distributions]
    stds = [np.std(arr) for arr in rl_distributions]

    means_random = [np.mean(arr) for arr in random_distributions]
    stds_random = [np.std(arr) for arr in random_distributions]

    x = np.arange(len(years))

    fig, ax = plt.subplots()

    # Define bar width and offsets
    width = 0.15
    offsets = {
        'Ceres': -2 * width,
        'Umbrella': -1 * width,
        'Zero': 0,
        'RL': 1 * width,
        'Random': 2 * width,
    }
    # Bars for baselines
    ax.bar(x + offsets['Ceres'], baseline_c, width, label='Ceres', color=cmap(2), alpha=alpha)
    ax.bar(x + offsets['Umbrella'], baseline_u, width, label='Umbrella', color=cmap(1), alpha=alpha)
    ax.bar(x + offsets['Zero'], baseline_z, width, label='Zero', color=cmap(0), alpha=alpha)

    # Bars with errorbars for distributions
    ax.bar(
        x + offsets['RL'],
        means,
        width,
        yerr=stds,
        capsize=5,
        label='RL (mean ± std)',
        color=cmap(3),
        alpha=alpha
    )
    ax.bar(
        x + offsets['Random'],
        means_random,
        width,
        yerr=stds_random,
        capsize=5,
        label='Random (mean ± std)',
        color=cmap(4),
        alpha=alpha
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.linspace(0,15,16))
    ax.set_ylabel('Pesticide Use')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()