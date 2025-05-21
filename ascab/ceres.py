import os
import argparse

import pandas as pd
from ascab.env.env import MultipleWeatherASCabEnv, get_weather_library, get_default_start_of_season, get_default_end_of_season, ActionConstrainer
from ascab.utils.generic import get_dates
from ascab.train import CeresOptimizer

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-y", "--year", default=2016, type=int)

    args = argparser.parse_args()

    country = 'FR'

    y = args.year

    ascab_env = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[(42.1620, 3.0924)] if country != 'FR' else [(44.0986, 1.1628)],
            dates=get_dates([y], start_of_season=get_default_start_of_season(),
                            end_of_season=get_default_end_of_season())),
        biofix_date="March 10",
        budbreak_date="March 10",
        mode="sequential",
    )
    ascab_env_constrained = ActionConstrainer(ascab_env)
    optimizer = CeresOptimizer(ascab_env_constrained,
                               os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                            "ceres",
                                            f"ceres_{y}_{country}.txt"))
    optimizer.run_optimizer()
    ceres_results = optimizer.run_ceres_agent()
    print(ceres_results)


if __name__ == '__main__':
    main()