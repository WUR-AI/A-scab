import datetime
import numpy as np
import pandas as pd

from ascab.utils.weather import compute_leaf_wetness_duration, is_wet
from ascab.utils.generic import parse_date
from ascab.model.infection import get_pat_threshold


def get_default_budbreak_date():
    return 'April 1'


def pseudothecial_development_has_ended(stage: np.float64) -> bool:
    """
    Determines whether pseudothecial development has ended, based on the presence of first mature ascospores.
    According to Figure 2 of Rossi et al. (p302), pseudothecial development is considered ended
    when the stage value is greater than or equal to 9.5.

    Parameters:
    - stage (float): The current stage value representing the developmental stage of pseudothecia.

    Returns:
    - bool: True if the stage value is greater than or equal to 9.5, indicating that pseudothecial
            development has ended and first mature ascospores are present; False otherwise.
    """
    return stage >= 9.5


def pat(dhw: np.float32) -> float:
    """
    Computes the proportion of seasonal ascospores that can potentially become airborne on a given day (PAT).

    This function uses equation 5 from Rossi et al., p. 302

    Parameters:
    - dhw (float): Degree hours of wetness.

    Returns:
    - float: The proportion of ascospores that can become airborne.
    """

    return 1.0 / (1.0 + np.exp(6.89 - 0.035 * dhw))


def get_dwh(pat_value: np.float32) -> float:
    return (np.log((1.0 / pat_value) - 1.0)-6.89) / -0.035


class PseudothecialDevelopment:
    """
    A class to model and update the pseudothecial development rate based on weather conditions.
    This class implements the equations (1-4) as described on page 301 of Rossi et al.

    Attributes:
    - value (float): The current value representing the stage of pseudothecial development, initialized to 5.0 by default.
    - rate (float): The rate of change in pseudothecial development, calculated based on daily weather data.

    Methods:
    - update_rate(df_weather_day: pd.DataFrame) -> float:
      Updates the development rate based on daily weather data.
    - compute_rate(current_value: np.float32, day: int, avg_temperature: np.float32, total_rain: np.float32,
                   humid_hours: int, wetness_duration: np.int64) -> float:
      Computes the rate of change in pseudothecial development using specified parameters.
    - integrate() -> None:
      Integrates the rate of change into the current value to update the development stage.
    """

    def __init__(self, start_day: str = "February 1", initial_value: np.float32 = 5.0):
        """
        Initializes the PseudothecialDevelopment class with an initial value.

        Parameters:
        - initial_value (np.float32): The initial value representing the stage of pseudothecial development. Defaults to 5.0.
        """
        super(PseudothecialDevelopment, self).__init__()
        self.value = initial_value
        self.rate = 0
        self.start_day = datetime.datetime.strptime(start_day, "%B %d").timetuple().tm_yday

    def update_rate(self, df_weather_day: pd.DataFrame):
        """
        Updates the development rate based on hourly weather data of a selected day.

        Parameters:
        - df_weather_day (pd.DataFrame): A DataFrame containing hourly weather data of a selected day

        Returns:
        - float: The updated rate of pseudothecial development.
        """
        day = df_weather_day.index.date[0].timetuple().tm_yday
        avg_temperature = df_weather_day['temperature_2m'].mean()
        total_rain = df_weather_day['precipitation'].sum()
        hours_humid = len(df_weather_day[df_weather_day['relative_humidity_2m'] > 85.0])
        wetness_duration = compute_leaf_wetness_duration(df_weather_day)
        self.rate = self.compute_rate(self.start_day, self.value, day, avg_temperature, total_rain, hours_humid, wetness_duration)
        return self.rate

    @staticmethod
    def compute_rate(start_day: int, current_value: np.float32, day: int, avg_temperature: np.float32, total_rain: np.float32,
                     humid_hours: int, wetness_duration: np.int64):
        """
        Computes the rate of change in pseudothecial development.

        This method calculates the daily change in pseudothecial development using weather parameters.
        It also checks certain conditions that might inhibit development, setting the rate to zero if any are met.

        Parameters:
        - start_day (int): Start of ontogenesis of pseudothecia
        - current_value (np.float32): The current value representing the stage of pseudothecial development.
        - day (int): The current day of the year.
        - avg_temperature (np.float32): The average temperature for the day in degrees Celcius.
        - total_rain (np.float32): The total rainfall for the day in mm.
        - humid_hours (int): The number of hours with relative humidity above 85%.
        - wetness_duration (np.int64): The duration of leaf wetness in hours.

        Returns:
        - float: The computed rate of change in pseudothecial development.
        """
        # Calculate the daily change in pseudothecial development (equation 1 Rossi et al. page 301)
        dy_dt = 0.0031 + 0.0546 * avg_temperature - 0.00175 * (avg_temperature ** 2)
        # Check conditions and modify dy_dt accordingly (equations 2 and 3 Rossi et al. page 301)
        condition = (day < start_day) or pseudothecial_development_has_ended(current_value) or (
                avg_temperature <= 0) or (total_rain <= 0.25) or (humid_hours <= 8) or (wetness_duration <= 8.0)
        dy_dt = np.where(condition, 0.0, dy_dt)
        return dy_dt

    def integrate(self):
        self.value += self.rate * 1.0


class AscosporeMaturation:
    """
    A class to model and update the maturation of ascospores based on weather conditions.
    This class implements the equations (5-8) as described on page 302 of Rossi et al.
    """

    def __init__(self, dependency: PseudothecialDevelopment, biofix_date=None):
        super(AscosporeMaturation, self).__init__()
        self.value = 0.0
        self.rate = 0.0
        self._dhw = 0
        self._delta_dhw = 0
        self._dependencies = dependency
        self.biofix_date = parse_date(biofix_date)

    def update_rate(self, df_weather_day: pd.DataFrame) -> np.float32:
        precipitation = df_weather_day['precipitation'].values
        vapour_pressure_deficit = df_weather_day['vapour_pressure_deficit'].values
        temperature_2m = df_weather_day['temperature_2m'].values
        day = df_weather_day.index.date[0].timetuple().tm_yday

        if (self.biofix_date is not None and day >= self.biofix_date):
            self._dhw = max(self._dhw, round(get_dwh(get_pat_threshold())))
            self.value = max(self.value, get_pat_threshold())
        if (self.biofix_date is not None and day >= self.biofix_date) or pseudothecial_development_has_ended(self._dependencies.value):
            self.rate, self._delta_dhw = self.compute_rate(np.float32(self._dhw), precipitation, vapour_pressure_deficit, temperature_2m)
        else:
            self.rate, self._delta_dhw = 0, 0
        return self.rate

    @staticmethod
    def compute_rate(current_dhw: np.float32, precipitation: np.ndarray[1, np.float32],
                     vapour_pressure_deficit: np.ndarray[1, np.float32], temperature_2m: np.ndarray[1, np.float32]) -> (np.float32, np.float32):
        wet_hourly = is_wet(precipitation, vapour_pressure_deficit)
        hw = wet_hourly * temperature_2m / float(len(wet_hourly))
        dhw = np.sum(hw)
        current_value = pat(current_dhw)
        new_value = pat(current_dhw + dhw)
        delta_value = new_value - current_value
        delta_dhw = dhw
        return delta_value, delta_dhw

    def integrate(self):
        self.value += self.rate * 1.0
        self._dhw += self._delta_dhw * 1.0


class LAI:
    """
    A class to model and update the Leaf Area Index of an apple tree.
    This class implements the description given on p305 of Rossi et al. with some modifications
    """
    def __init__(self, start_date: str = get_default_budbreak_date()):
        super(LAI, self).__init__()
        self.value = 0
        self.rate = 0
        self.start_date = parse_date(start_date)

    def update_rate(self, df_weather_day: pd.DataFrame):
        day = df_weather_day.index.date[0].timetuple().tm_yday
        avg_temperature = df_weather_day['temperature_2m'].mean()
        self.rate = self.compute_rate(self.start_date, np.float32(self.value), day, avg_temperature)
        return self.rate

    @staticmethod
    def compute_rate(start_day: int, current_value: np.float32, day: int, avg_temperature: np.float32):
        # Calculate the daily change
        # TODO: Definition in Rossi not trivial. We may need to incorporate number of shoots
        number_of_shoots_per_m2 = 85
        dy_dt = 0.00008 * max(0.0, (avg_temperature - 4.0)) * number_of_shoots_per_m2
        # Check conditions and modify dy_dt accordingly
        condition = (day < start_day) or (current_value > 5.0)
        dy_dt = np.where(condition, 0.0, dy_dt)
        return dy_dt

    def integrate(self):
        self.value += self.rate * 1.0


def map_temperature_to_bbch(temperature_sum: float) -> float:
    # Fixed points for BBCH scale
    # Stoeckli, S., & Samietz, J. Simplified modelling of Apple flowering phenology for application in climate change scenarios.
    # IX International Symposium on Modelling in Fruit Research and Orchard Management 1068 (pp. 153-160).
    bbch_values = np.array([60, 65, 69], dtype=np.float32)
    temp_sums = np.array([10368, 12546, 15914], dtype=np.float32)

    # Return zero if the temperature sum is outside the known range
    if temperature_sum < temp_sums[0] or temperature_sum > temp_sums[-1]:
        return 0
    
    # Linearly interpolate the BBCH value for the given temperature sum
    bbch_value = np.interp(temperature_sum, temp_sums, bbch_values)
    
    return bbch_value


class Phenology:
    def __init__(self, start_date=45, base_temperature: float = 0.0):
        super(Phenology, self).__init__()
        self.value = 0
        self.temperature_sum = 0.0
        self.rate = 0.0
        self.start_date = parse_date(start_date)
        self.base_temperature = base_temperature

    def update_rate(self, df_weather_day: pd.DataFrame):
        day = df_weather_day.index.date[0].timetuple().tm_yday
        temperature_hourly = df_weather_day['temperature_2m']
        self.rate = self.compute_rate(self.start_date, self.base_temperature, day, temperature_hourly)
        return self.rate

    @staticmethod
    def compute_rate(start_day: int, base_temperature: float, day: int, temperature_hourly: np.ndarray[1, np.float32],):
        # Calculate the daily change
        temperature_diff = np.maximum(0, temperature_hourly - base_temperature)
        # Sum over the 24 hours
        dy_dt = np.sum(temperature_diff)
        # Check conditions and modify dy_dt accordingly
        condition = (day < start_day)
        dy_dt = np.where(condition, 0.0, dy_dt)
        return dy_dt

    def integrate(self):
        self.temperature_sum += self.rate * 1.0
        self.value = map_temperature_to_bbch(self.temperature_sum)
