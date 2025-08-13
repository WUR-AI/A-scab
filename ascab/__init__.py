import os
import gymnasium as gym

from .env.env import MultipleWeatherASCabEnv, ActionConstrainer, get_weather_library_from_csv

do_check_env = False
if do_check_env:
    from gymnasium.utils.env_checker import check_env
    check_env(MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library_from_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", f"train.csv")),
        biofix_date="March 10",
        budbreak_date="March 10",
        discrete_actions=True,
        truncated_observations='truncated',
    ))

def _wrapper_picker(wrapper):
    if wrapper is None:
        return ()
    elif wrapper == 'ConditionalAgents':
        return (
            ActionConstrainer.wrapper_spec(
                risk_period=True
            ),
        )

def _register_ascab_env(dataset: str = 'train',
                        is_discrete: bool = True,
                        use_wrapper: str = None,
                        competition_name: str = "test-competition-232675",):

    weather_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hackathon", "dataset", f"{dataset}.csv")
    library = get_weather_library_from_csv(weather_path)

    str_discrete = "Discrete" if is_discrete else "Continuous"
    str_dataset = dataset.title()

    if use_wrapper == "ConditionalAgents":
        str_use_wrapper = "-NonRL"
    elif use_wrapper == "Penalty":
        str_use_wrapper = "-Pen"
    elif use_wrapper == "ConditionalAgentsPenalty":
        str_use_wrapper = "-NonRLPen"
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

print(f"Please wait a few seconds; importing the AscabGym modules...")
for data in ['train', 'val']:
    for discrete in [True, False]:
        _register_ascab_env(dataset=data, is_discrete=discrete, use_wrapper=None)
print(f"Imported successfully!")