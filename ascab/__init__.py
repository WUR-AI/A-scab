import os
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from .env.env import MultipleWeatherASCabEnv, PenaltyWrapper, ActionConstrainer, get_weather_library_from_csv

# check_env(MultipleWeatherASCabEnv(
#     weather_data_library=get_weather_library_from_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", f"train.csv")),
#     biofix_date="March 10",
#     budbreak_date="March 10",
#     discrete_actions=True,
#     truncated_observations='truncated'
# ))

def _wrapper_picker(wrapper):
    if wrapper is None:
        return ()
    elif wrapper == 'ConditionalAgents':
        return (
            ActionConstrainer.wrapper_spec(
                risk_period=True
            ),
        )
    elif wrapper == "Penalty":
        return(
            PenaltyWrapper.wrapper_spec(
                penalty=0.025
            )
        )
    elif wrapper == "ConditionalAgentsPenalty":
        return (
            ActionConstrainer.wrapper_spec(
                risk_period=True
            ),
            PenaltyWrapper.wrapper_spec(
                penalty=0.025
            )
        )

def _register_ascab_env(dataset: str = 'train',
                        is_discrete: bool = True,
                        use_wrapper: str = None,
                        competition_name: str = "test-competition-232675",):

    weather_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", f"{dataset}.csv")

    if not os.path.exists(weather_path):
        weather_path = f"/kaggle/input/{competition_name}/{dataset}.csv"


    library = get_weather_library_from_csv(weather_path)

    str_discrete = "Discrete" if is_discrete else "Continuous"
    str_dataset = dataset.title()

    if use_wrapper == "ConditionalAgents":
        str_use_wrapper = "-NonRLNoPenalty"
    elif use_wrapper == "Penalty":
        str_use_wrapper = "-Pen"
    elif use_wrapper == "ConditionalAgentsPenalty":
        str_use_wrapper = "-NonRL"
    else:
        str_use_wrapper = ""

    gym.register(
        id=f'Ascab{str_dataset}Env-{str_discrete}'+str_use_wrapper,
        entry_point='ascab.env.env:MultipleWeatherASCabEnv',
        kwargs={
            "weather_data_library": library,
            "biofix_date": "March 10",
            "budbreak_date": "March 10",
            "discrete_actions": is_discrete,
            "truncated_observations": "truncated",
            "mode": 'sequential' if dataset == 'val' else 'random',
        },
        additional_wrappers=_wrapper_picker(wrapper=use_wrapper),
    )

for data in ['train', 'val']:
    for discrete in [True, False]:
        for wrapper in [None, 'Penalty', 'ConditionalAgentsPenalty', 'ConditionalAgents']:
            _register_ascab_env(dataset=data, is_discrete=discrete, use_wrapper=wrapper)