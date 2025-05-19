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
    return [
        "Precipitation",
         "AscosporeMaturation",
         "Discharge",
         "Pesticide",
         "Risk",
         "Action",
    ]


def plot_results(results: [Union[dict[str, pd.DataFrame], pd.DataFrame]],
                 variables: list[str] = get_default_plot_variables(),
                 save_path: str = None,
                 fig_size: int = 10,
                 save_type: str = 'png',
                 per_year: bool = False,
                 zoom: bool = False,
                 stacked: bool = False,):
    results = {"": results} if not isinstance(results, dict) else results
    alpha = 1.0 if len(results) == 1 else 0.6

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
            plt.savefig(save_path, format=save_type, dpi=600, bbox_inches='tight')
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    else:  # if year is True
        print("Printing per year results!~")
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
                    out_path = os.path.join(save_path, f"plot_{year}.png")
                    print(f'save {out_path}')
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=600)
                plt.show()
                plt.close(fig)
        elif zoom and stacked:
            print("Printing stacked zoomed results!~")
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

                _, axes_zoom = make_year_plot(year, results, num_variables, fig_size=fig_size, alpha=alpha,
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

                        # coords = {'p1': ([start_date, ])}
                        # for k, s, e in coords.items():
                        #     fig_combined.add_artist(
                        #         Line2D([0.02, 0.98], [0.05, 0.95],  # x,y in Figure coords (0-1)
                        #                transform=fig_combined.transFigure,  # <─ key: use Figure coords
                        #                color=color_line, lw=1, ls="--", zorder=0)
                        #     )

                if save_path:
                    out_path = os.path.join(save_path, f"plot_{year}_stacked.png")
                    print(f'save {out_path}')
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=600)
                # plt.tight_layout()
                plt.show()
                plt.close(fig_combined)

        else:  # if zoom is True:
            print("Printing zoomed results!~")
            for year in sorted(set().union(*(df["Year"].unique()
                                           for df in results.values()))):
                fig, axes = make_year_plot(year, results, num_variables, fig_size=10, alpha=alpha)
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
                    plt.savefig(out_path, bbox_inches="tight", format=save_type, dpi=600)
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


def make_year_plot(year, results_dict, num_variables=6, fig_size=9, alpha=0.5, stacked=False, container=None):
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

def plot_normalized_reward(dict_extracted, baselines_extracted, random_extracted, save_path: str = None):
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

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1,11))
    ax.set_ylabel('Normalized Reward based on Ceres')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, axis='y')

    if save_path:
        out_path = os.path.join(save_path, f"plot_reward.png")
        print(f'save {out_path}')
        plt.savefig(out_path, bbox_inches="tight", format='png', dpi=600)

    plt.tight_layout()
    plt.show()

def plot_pesticide_use(dict_extracted, baselines_extracted, random_extracted, pareto_line: bool = False, save_path=None,
                       avg_line: bool = False):
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

    if pareto_line:
        ax1 = pareto_line_plot(x, baseline_c, baseline_u, means, offsets, ax, cmap, alpha)

    if avg_line:
        plot_avg_line(alpha, ax, baseline_c, baseline_u, baseline_z, cmap, means, means_random)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.linspace(0,15,16))
    ax.set_ylabel('Pesticide Use per year')
    ax.set_xlabel('Year')
    if pareto_line:
        ax1.set_ylabel('Cumulative Pesticide Use')
        ax1.set_yticks(np.linspace(0, 50, 11))
    ax.legend()
    # ax.grid(True, axis='y')

    if save_path:
        out_path = os.path.join(save_path, f"plot_pesticide_use.png")
        print(f'save {out_path}')
        plt.savefig(out_path, bbox_inches="tight", format='png', dpi=600)

    plt.tight_layout()
    plt.show()


def plot_avg_line(alpha, ax, baseline_c, baseline_u, baseline_z, cmap, means, means_random):
    ax.axhline(np.median(baseline_c), alpha=alpha, color=cmap(2), linestyle='--')
    ax.axhline(np.median(baseline_u), alpha=alpha, color=cmap(1), linestyle='--')
    ax.axhline(np.median(baseline_z), alpha=alpha, color=cmap(0), linestyle='--')
    ax.axhline(np.median(means), alpha=alpha, color=cmap(3), linestyle='--')
    ax.axhline(np.median(means_random), alpha=alpha, color=cmap(4), linestyle='--')
    return ax


def pareto_line_plot(x, baseline_c, baseline_u, means, offsets, ax, cmap, alpha):
    cumsum_ceres = np.cumsum(baseline_c)
    cumsum_umb = np.cumsum(baseline_u)
    # cumsum_zro = np.cumsum(baseline_z)
    cumsum_rl = np.cumsum(means)
    # cumsum_random = np.cumsum(means_random)
    kwargs_scatter = {'marker': 'o', 'markeredgecolor': 'black'}
    ax1 = ax.twinx()
    ax1.plot(x + offsets['Ceres'], cumsum_ceres, color=cmap(2), alpha=alpha, **kwargs_scatter)
    ax1.plot(x + offsets['Umbrella'], cumsum_umb, color=cmap(1), alpha=alpha, **kwargs_scatter)
    # ax1.plot(x + offsets['Zero'], cumsum_zro, color=cmap(0), alpha=alpha, **kwargs_scatter)
    ax1.plot(x + offsets['RL'], cumsum_rl, color=cmap(3), alpha=alpha, **kwargs_scatter)
    # ax1.plot(x + offsets['Random'], cumsum_random, color=cmap(4), alpha=alpha, **kwargs_scatter)
    return ax1


def plot_risk(dict_extracted, baselines_extracted, random_extracted, pareto_line: bool = False, save_path: str = None,
              avg_line: bool = False):
    years = sorted(dict_extracted['Risk'].keys())
    cmap = plt.get_cmap('tab10')
    alpha = 0.9

    baseline_u = []  # Umbrella
    baseline_z = []  # Zero
    baseline_c = []
    random_distributions = []
    rl_distributions = []

    for yr in years:
        rl_raw = np.array(dict_extracted['Risk'][yr])
        random_raw = np.array(random_extracted['Risk'][yr])


        # baselines_extracted['Reward'][yr] == [ceres, umbrella, zero]
        ceres, umb, zro = baselines_extracted['Risk'][yr]
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

    if pareto_line:
        ax1 = pareto_line_plot(x, baseline_c, baseline_u, means, offsets, ax, cmap, alpha)

    if avg_line:
        plot_avg_line(alpha, ax, baseline_c, baseline_u, baseline_z, cmap, means, means_random)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, .2)
    ax.set_yticks(np.linspace(0,.2 ,11))
    ax.set_ylabel('Risk per year')
    ax.set_xlabel('Year')
    if pareto_line:
        ax1.set_ylabel('Cumulative Risk')
        # ax1.set_yticks(np.linspace(0,50,11))
    ax.legend(loc='upper left')
    # ax.grid(True, axis='y')

    if save_path:
        out_path = os.path.join(save_path, f"plot_risk.png")
        print(f'save {out_path}')
        plt.savefig(out_path, bbox_inches="tight", format='png', dpi=600)

    plt.tight_layout()
    plt.show()


def plot_use_vs_risk(dict_extracted, baselines_extracted, random_extracted):
    """
    Relationship-first view: every dot is one algorithm in one year.
    Years are colour-coded so the path through time is visible.
    Error bars (1 SD) are drawn for RL and Random, where you have distributions.
    """

    years = sorted(dict_extracted['Pesticide'].keys())
    cmap = plt.get_cmap('tab10')

    # --- Collect the numbers -------------------------------------------------
    # Baselines arrive as single values
    ceres_use, ceres_risk   = [], []
    umb_use,   umb_risk     = [], []
    zero_use,  zero_risk    = [], []

    # RL / Random arrive as distributions → need mean & SD
    rl_use_mean, rl_use_sd       = [], []
    rl_risk_mean, rl_risk_sd     = [], []
    rnd_use_mean, rnd_use_sd     = [], []
    rnd_risk_mean, rnd_risk_sd   = [], []

    for yr in years:
        # baselines_extracted['Pesticide'][yr] == [ceres, umbrella, zero]
        cu, uu, zu = baselines_extracted['Pesticide'][yr]
        cr, ur, zr = baselines_extracted['Risk'][yr]

        ceres_use.append(cu);   ceres_risk.append(cr)
        umb_use.append(uu);     umb_risk.append(ur)
        zero_use.append(zu);    zero_risk.append(zr)

        # RL + Random – take stats of the raw arrays
        rl_u  = np.asarray(dict_extracted['Pesticide'][yr])
        rl_r  = np.asarray(dict_extracted['Risk'][yr])
        # rnd_u = np.asarray(random_extracted['Pesticide'][yr])
        # rnd_r = np.asarray(random_extracted['Risk'][yr])

        rl_use_mean.append(rl_u.mean());     rl_use_sd.append(rl_u.std())
        rl_risk_mean.append(rl_r.mean());    rl_risk_sd.append(rl_r.std())
        # rnd_use_mean.append(rnd_u.mean());   rnd_use_sd.append(rnd_u.std())
        # rnd_risk_mean.append(rnd_r.mean());  rnd_risk_sd.append(rnd_r.std())

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    sc_kwargs = dict(s=65, edgecolors='k', linewidths=.4)   # consistent look
    errbar_marker_kws = dict(markeredgecolor='k',
                             markeredgewidth=.4,
                             markersize=7,
                             linestyle='none')

    # Each baseline: one marker per year, coloured by year
    points = ax.scatter(ceres_use, ceres_risk, color=cmap(2),
                        marker='o', label='Ceres',   **sc_kwargs)
    ax.scatter(umb_use, umb_risk, color=cmap(1),
               marker='o', label='Umbrella', **sc_kwargs)
    # ax.scatter(zero_use,  zero_risk, color=cmap(0),
    #            marker='^', label='Zero',  **sc_kwargs)

    # RL & Random with 1-SD error bars
    ax.errorbar(rl_use_mean,  rl_risk_mean,  xerr=rl_use_sd,  yerr=rl_risk_sd,
                fmt='o', capsize=4, color=cmap(3), label='RL (mean ± sd)',   **errbar_marker_kws)
    # ax.errorbar(rnd_use_mean, rnd_risk_mean, xerr=rnd_use_sd, yerr=rnd_risk_sd,
    #             fmt='v', capsize=4, label='Random (mean ± sd)', **errbar_marker_kws)

    # Baselines
    for x, y, yr in zip(ceres_use, ceres_risk, years):
        ax.annotate(str(yr), (x, y), xytext=(4, 4), textcoords='offset points',
                    ha='left', va='bottom', fontsize=8)

    for x, y, yr in zip(umb_use, umb_risk, years):
        ax.annotate(str(yr), (x, y), xytext=(4, 4), textcoords='offset points',
                    ha='left', va='bottom', fontsize=8)

    # for x, y, yr in zip(zero_use, zero_risk, years):
    #     ax.annotate(str(yr), (x, y), xytext=(4, 4), textcoords='offset points',
    #                 ha='left', va='bottom', fontsize=8)

    # RL / Random – annotate the *means*
    for x, y, yr in zip(rl_use_mean, rl_risk_mean, years):
        ax.annotate(str(yr), (x, y), xytext=(4, 4), textcoords='offset points',
                    ha='left', va='bottom', fontsize=8)

    # for x, y, yr in zip(rnd_use_mean, rnd_risk_mean, years):
    #     ax.annotate(str(yr), (x, y), xytext=(4, 4), textcoords='offset points',
    #                 ha='left', va='bottom', fontsize=8)

    # Optional: LOWESS or linear trend across ALL points
    # all_use  = np.concatenate([ceres_use, umb_use, zero_use,
    #                            rl_use_mean, rnd_use_mean])
    # all_risk = np.concatenate([ceres_risk, umb_risk, zero_risk,
    #                            rl_risk_mean, rnd_risk_mean])
    # sns.regplot(x=all_use, y=all_risk, scatter=False,
    #             lowess=True, ci=None, ax=ax, line_kws={'lw':1.4, 'ls':'--'})

    # ------------------------------------------------------------------

    # Cosmetics
    # cbar = fig.colorbar(points, ax=ax, pad=.02)
    # cbar.set_label('Year')

    ax.set_xlabel('Total pesticide use')
    ax.set_ylabel('Total risk index')
    ax.grid(True, ls=':')
    ax.legend()
    plt.tight_layout()
    plt.show()