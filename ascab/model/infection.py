import numpy as np
import pandas as pd
import pytz
import datetime
from typing import Optional

from ascab.utils.weather import is_rain_event, compute_duration_and_temperature_wet_period
from ascab.utils.generic import items_since_last_true


def get_pat_threshold() -> float:
    return 0.016


def determine_discharge_hour_index(pat: float, rain_events: np.ndarray[1, np.bool_], is_daytime: np.ndarray[1, np.bool_],
                                   pat_previous: float, hours_since_previous: np.ndarray[1, np.int_])\
                                    -> Optional[np.ndarray[1, np.int_]]:
    """
    Determines the index of the hour at which ascospores are discharged based on various conditions.

    This function implements the section on Ascospore discharge events in Rossi et al. page 303-304.
    The function returns the index of the hour when discharge occurs or `None` if no discharge is determined.

    Parameters:
    - pat (float): The current PAT value.
    - rain_events (np.ndarray[np.bool_]): A 1-dimensional NumPy array indicating rain events, where `True` represents
      a rain event and `False` represents no rain.
    - is_daytime (np.ndarray[np.bool_]): A 1-dimensional NumPy array indicating daytime status, where `True` represents
      daytime and `False` represents nighttime.
    - pat_previous (float): The PAT value from the previous time period.
    - hours_since_previous (np.ndarray[np.int_]): A 1-dimensional NumPy array indicating the number of hours since the
      previous discharge event.

    Returns:
    - Optional[int]: The index of the hour when discharge occurs. Returns `None` if no discharge is determined based on
      the provided conditions.
    """

    # After PAT ′has reached pat_threshold, daytime rain events cause the instantaneous discharge of mature ascospores
    # so that they begin to be airborne immediately
    if np.logical_or(pat <= get_pat_threshold(), 1 not in rain_events):
        return None

    first_rain_hour = np.where(rain_events == 1)[0][0]
    heavy_dew = False
    # Rossi et al. page 304 (top left):
    # A rain event does not lead to spore ejection when:
    # * it occurs after less than 5 h from the preceding discharge event
    # * leaf litter dries before 7.00 am following a rain event in night time when PAT < 0.80 [Not implemented]
    # * nightly rainfalls are followed by heavy dew deposition that persists some hours after sunrise
    if np.logical_or(hours_since_previous[first_rain_hour] < 5.0, heavy_dew):
        return None
    # no delay
    # * PAT ≥ 0.80
    # * more than one third of the total season’s ascospores is mature inside pseudothecia
    #   (SRAdis ≥ 0.30, where SRAdis is the part of PAT becoming airborne on the discharge event).
    if np.logical_or(np.logical_or(pat >= 0.80, (pat - pat_previous) >= 0.30), is_daytime[first_rain_hour]):
        return first_rain_hour

    # delay: sunset
    first_daytime = np.where(is_daytime == 1)[0][0]
    return first_daytime


def compute_ds1(temperature: float, hour_since_onset: np.ndarray[1, np.int_]):
    """
    Computes the ascospore dose ejected over time based on temperature and hours since onset.

    This function uses the formula from Rossi et al., page 304, equation 11, to calculate the ascospore dose
    as a function of temperature and the number of hours since onset. The result is a NumPy array containing
    the computed dose for each hour since onset. Hours that are negative are set to 0 in the resulting array.

    Parameters:
    - temperature (float): The temperature in degrees Celsius, averaged over the duration of a discharge event
    - hour_since_onset (np.ndarray[np.int_]): A 1-dimensional NumPy array containing the number of hours since onset.
      Each element should be an integer representing the time in hours.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_onset`, where each element
      represents the computed ascospore dose for the corresponding hour since onset. Hours that are negative in the
      input array will result in a dose of 0 in the output array.
    """

    result = np.array(1.0 / (1.0 + np.exp(2.999 - 0.067 * temperature * hour_since_onset)))
    mask = hour_since_onset < 0
    result[mask] = 0
    return result


def compute_derivative_ds1(temperature: float, hour_since_onset: np.ndarray[1, np.int_]) -> np.ndarray[1, np.float64]:
    """
    Computes the derivative of the ascospore dose ejection function (Rossi et al., page 304, equation 11)

    Parameters:
    - temperature (float): The temperature in degrees Celsius, averaged over the duration of a discharge event
    - hour_since_onset (np.ndarray[np.int_]): A 1-dimensional NumPy array containing the number of hours since onset.
      Each element should be an integer representing the time in hours.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_onset`, where each element
      represents the derivative of the computed ascospore dose for the corresponding hour since onset.
    """

    u = 2.999 - 0.067 * temperature * hour_since_onset
    result = 0.067 * temperature * np.exp(u) / ((1.0 + np.exp(u)) ** 2)
    # Note that we omit the clipping done in compute_ds1 in order to arrive at the same ds1 value at onset

    return result


def compute_sdl_wet(rain: np.ndarray[1, np.float64], lai: float = 1.0, height: float = 0.5) -> np.ndarray[1, np.float64]:
    """
    Computes wet deposition of ascospores (Rossi et al., page 304, equations 13-14)

    Parameters:
    - rain (np.ndarray[np.float64]): Hourly rainfall in mm
    - lai (float): Leaf area index
    - height (float): Height above the ground of the lowest leaves in m

    Returns:
    - np.ndarray: Part of ascospores deposited on apple leaves deposited by wet deposition

    See also Rossi et al. IOBC/WPRS Bulletin 29, 53–58.
    """

    lambda_h = (1.0 / (1.0 + np.exp(2.575 - 0.987 * lai * (5.022 * (rain ** 0.063)))))
    result = (1.017 * 0.374 ** height) * lambda_h
    return result


def compute_sdl_dry(lai: float = 1.0) -> float:
    """
    Computes dry deposition of ascospores (Rossi et al., page 304, equation 15)

    Parameters:
    - lai (float): Leaf area index

    Returns:
    - float: Part of ascospores deposited on apple leaves deposited by dry deposition
    """

    result = 0.594 - (0.643 * 0.372 ** lai)
    return result


def compute_deposition_rate(rain: np.ndarray[1, np.float64], lai: float = 1.0, height: float = 0.5, do_clip: bool = True):
    """
    Computes proportion of ascospores deposited on apple leaves (Rossi et al., page 304, equation 12)

    Parameters:
    - rain (np.ndarray[np.float64]): Hourly rainfall in mm
    - lai (float): Leaf area index
    - height (float): Height above the ground of the lowest leaves in m
    - do_clip (bool):  Clip to ensure that the fraction remains between 0 and 1, as it may otherwise exceed this range.

    Returns:
    - np.ndarray[np.float64]: Part of ascospores deposited on apple leaves deposited
    """

    ds_wet = compute_sdl_wet(rain, lai, height)  #[0.04-0.622]
    ds_dry = compute_sdl_dry(lai)  #max 0.594
    ds_sum = ds_wet + ds_dry
    if do_clip:
        ds_sum = np.clip(ds_sum, None, 1.0)
    return ds_sum


def compute_ds2(temperature: float, hour_since_onset: np.ndarray[1, np.int_]) -> np.ndarray[1, np.float64]:
    """
    Computes fraction of ascospores in stage2 (i.e. germinated), following Rossi et al., page 305, equations 16-17

    This function uses the formulas from Rossi et al., page 305, equations 16-17, to calculate the fraction of ascospore
    that has reached stage 2 (i.e. germinated), as a function of temperature and the number of hours since onset.

    Parameters:
    - temperature (float): The temperature in degrees Celsius, averaged over the duration of an infection event
    - hour_since_onset (np.ndarray[np.int_]): A 1-dimensional NumPy array containing the number of hours since onset.
      Each element should be an integer representing the time in hours.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_onset`, where each element
      represents the fraction of ascospore that has reached stage 2
    """

    result_below_20 = 1.0 / (1.0 + np.exp((5.23 - 0.1226 * temperature + 0.0014 * (temperature ** 2)) - (
            0.093 + 0.0112 * temperature - 0.000122 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.97 + 0.4297 * temperature - 0.0061 * (temperature ** 2)) - (
            0.416 - 0.0031 * temperature - 0.000245 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_ds3(temperature: float, hour_since_onset: np.ndarray[1, np.int_]) -> np.ndarray[1, np.float64]:
    """
    Computes fraction of ascospores in stage3 (i.e. with appressorium), following Rossi et al., page 305, equations 18-19

    This function uses the formulas from Rossi et al., page 305, equations 18-19, to calculate the fraction of ascospore
    that has reached stage 3 (i.e. with appressorium), as a function of temperature and the number of hours since onset.

    Parameters:
    - temperature (float): The temperature in degrees Celsius, averaged over the duration of an infection event
    - hour_since_onset (np.ndarray[np.int_]): A 1-dimensional NumPy array containing the number of hours since onset.
      Each element should be an integer representing the time in hours.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_onset`, where each element
      represents the fraction of ascospore that has reached stage 3
    """

    result_below_20 = 1.0 / (1.0 + np.exp((6.33 - 0.0647 * temperature - 0.000317 * (temperature ** 2)) - (
            0.111 + 0.01240 * temperature - 0.000181 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.13 + 0.5302 * temperature - 0.009130 * (temperature ** 2)) - (
            0.405 + 0.00079 * temperature - 0.000347 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_ds1_mor(hour_since_last_rain: np.ndarray[1, np.int_]):
    """
    Computes mortality rate of ascospores in stage1 (Rossi et al., page 305, equation 20)

    We assume here that the computed fraction is not a cumulative fraction

    Parameters:
    - hour_since_last_rain (np.ndarray[np.int_]): A 1-dimensional NumPy array with the number of hours since last rain.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_last_rain`, where each element
      represents the fraction of ascospore that has died
    """

    result = 0.263 * (1.0 - 0.97315 ** hour_since_last_rain)
    return result


def compute_ds2_mor(hour_since_last_rain: np.ndarray[1, np.int_], temperature: np.ndarray[1, np.float64],
                    humidity: np.ndarray[1, np.float64]):
    """
    Computes mortality rate of ascospores in stage2 (Rossi et al., page 305, equation 21)

    We assume here that the computed fraction is not a cumulative fraction

    Parameters:
    - hour_since_last_rain (np.ndarray[np.int_]): A 1-dimensional NumPy array with the number of hours since last rain.
    - temperature (np.ndarray[np.float64]): A 1-dimensional NumPy array with the temperature in degrees Celcius.
    - humidity (np.ndarray[np.float64]): A 1-dimensional NumPy array with the humidity as a percentage

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_last_rain`, where each element
      represents the fraction of ascospore that has died
    """

    result = (-1.538 + 0.253 * temperature - 0.00694 * (temperature ** 2)) * \
             (1 - 0.977 ** hour_since_last_rain) * (0.0108 * humidity - 0.008)
    return result


def compute_ds3_mor(hour_since_last_rain, temperature):
    """
    Computes mortality rate of ascospores in stage2 (Rossi et al., page 305, equation 22)

    We assume here that the computed fraction is not a cumulative fraction

    Parameters:
    - hour_since_last_rain (np.ndarray[np.int_]): A 1-dimensional NumPy array with the number of hours since last rain.
    - temperature (np.ndarray[np.float64]): A 1-dimensional NumPy array with the temperature in degrees Celcius.

    Returns:
    - np.ndarray: A 1-dimensional NumPy array of the same shape as `hour_since_last_rain`, where each element
      represents the fraction of ascospore that has died
    """

    result = (0.0028 * hour_since_last_rain) * \
             (-1.27 + 0.326 * temperature - 0.0102 * (temperature ** 2))
    return result


def compute_leaf_development(lai: float) -> float:
    """
    Computes leaf development (Rossi et al., page 305, equation 23)

    Parameters:
    - lai (float): Leaf area index

    Note that the computation looks rather suspicious, as the result becomes negative when lai exceeds 0.125
    """

    # TODO: looks suspicious
    result = 1 / (-5445.5 * (lai ** 2) + 661.55 * (lai))
    return result


def compute_delta_incubation(temperature: float) -> float:
    """
    Computes daily progress of incubation (Rossi et al., page 305, equation 25)

    Parameters:
    - temperature (float): Temperature
    """

    result = 1.0 / (26.4 - 1.0268 * temperature)
    return result


def get_discharge_date(df_weather_day: pd.DataFrame, pat_previous: float, pat_current: float,
                       time_previous: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Computes the discharge date following Rossi et al. Fig. 2 p302

    Parameters:
    - df_weather_day (pd.DataFrame): Weather
    - pat_previous (float): Value of "Proportion of ascospores that can become airborne" at the previous event
    - pat_previous (float): Proportion of ascospores that can become airborne at the current event
    - time_previous (pd.Timestamp): Start of the previous event

    Returns:
    - pd.Timestamp: Day of discharge
    """

    if pat_current > 0.99: return None
    # TODO: past 24 hours not taken into account
    rain_events = is_rain_event(df_weather_day)
    is_daytime = df_weather_day['is_day'].to_numpy()
    hours_since_previous = (df_weather_day.index - time_previous).total_seconds() / 3600
    discharge_hour_index = determine_discharge_hour_index(pat_current, rain_events, is_daytime, pat_previous,
                                                          hours_since_previous)
    if discharge_hour_index is None: return None
    discharge_date = df_weather_day.index[discharge_hour_index]
    return discharge_date


def meets_infection_requirement(temperature: float, wet_hours: int):
    """
    Determines whether requirements for an infection event are met, according to table 2 Stensvand et al. (1997)

    Parameters:
    - temperature (float): Average temperature in degrees Celsius during wet hours
    - wet_hours (int): Number of wet hours

    Returns:
    - np.bool: whether requirements for an infection event are met
    """

    infection_table = {
        'temperature': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        'ascospore': [40.5, 34.7, 29.6, 27.8, 21.2, 18.0, 15.4, 13.4, 12.2, 11.0,
                      9.0, 8.3, 8.0, 7.0, 7.0, 6.1, 6.0, 6.0, 6.0, 6.0,
                      6.0, 6.0, 6.0, 6.1, 8.0, 11.3],
        'conidia': [37.4, 33.6, 30.0, 26.6, 23.4, 20.5, 17.8, 15.2, 12.6, 10.0,
                    9.5, 9.3, 9.2, 9.2, 9.2, 9.0, 8.8, 8.5, 8.2, 7.9,
                    7.8, 7.8, 8.3, 9.3, 11.1, 14.0]
    }
    required_wet_hours = np.interp(temperature, infection_table['temperature'], infection_table['ascospore'])
    return wet_hours >= required_wet_hours


def will_infect(df_weather_infection: pd.DataFrame) -> (bool, int, float):
    """
    Determines whether requirements for an infection event are met, according to table 2 Stensvand et al. (1997)

    Parameters:
    - df_weather_infection (pd.DataFrame): Weather in next few days

    Returns:
    - bool: whether requirements for an infection event are met
    - int: duration of infection period
    - float: average temperature during infection
    """

    infection_duration, infection_temperature = compute_duration_and_temperature_wet_period(df_weather_infection)
    result = meets_infection_requirement(infection_temperature, infection_duration)
    return result, infection_duration, infection_temperature


class Discharge:
    def __init__(self, discharge_date: pd.Timestamp, ascospore_value: float):
        super(Discharge, self).__init__()
        self.discharge_date = discharge_date
        self.pat_start = ascospore_value


class InfectionRate:
    """
    Implementation of Infection model from Rossi et al., figure 2

    The model describes the progression of spores deposited on an apple tree.
    Spores advance through three successive stages, represented by sigmoids that indicate the proportion of spores at each stage.
    Spores may die; mortality is determined by weather conditions
    The risk of infection is determined by the number of spores that survive to the third stage.

    The original model seems a bit ambiguous at some points, especially eq. 24.
    The following assumptions are done:
    * Host susceptibility (HOST_inf in eq. 24) is ignored as computation of LAI development is questionable
      (i.e. eq. 23 yields negative values when LAI>0.125)
    * The length of the infection period is computed as the hours of wetness (t_inf) p304, starting at midnight
      Start of the infection is set to the discharge_hour (i.e. not start at midnight (!))
    * Risk for a whole infection period is calculated as follows:
      (1) determine the number of spores in stage 3 each hour
      (2) take the value at the end of the day (i.e. not at the end of the infection period, as Rossi may intend (!))
      (3) sum all end-of-day-values in infection period
    (!) Note that the sigmoids of stage3 may not have reached to 100% yet, as infection duration is not a variable in these sigmoids

    Mortality is computed as follows (for each hour):
      (1) The distribution of spores over the stages (i.e. fractions) is computed under a no-mortality assumption
      (2) The stage-specific mortality is computed
      (3) Stage-specific mortality (i.e. (2)) is multiplied with the fraction of that stage (i.e. (1)), resulting in total death
      (4) The population that died is subtracted from the population that has survived

    A similar approach is taken for the deposition of spores (for each hour):
      (1) Determine the fraction that will be released
      (2) Determine the fraction that will not be deposited
      (3) Multiply (1) and (2) to obtain the amount that will be lost; subtract from population that has survived

    Incubation period is not implemented (yet) as that won't influence end result
    """

    def __init__(self, discharge_date: pd.Timestamp, ascospore_value: float, previous_ascospore_value: float,
                 lai: float, duration: int, temperature: float):
        super(InfectionRate, self).__init__()
        self.discharge_date = discharge_date
        self.pat_start = ascospore_value
        self.pat_previous = previous_ascospore_value
        self.lai = lai

        self.infection_duration = duration
        self.infection_temperature = temperature

        self.incubation = []
        self.risk = []
        self.infection_efficiency = []

        self.hours = []
        self.s1_sigmoid = []
        self.s2_sigmoid = []
        self.s3_sigmoid = []
        self.s1 = []
        self.s2 = []
        self.s3 = []
        self.total_population = []
        self.mor0 = []
        self.mor1 = []
        self.mor2 = []
        self.mor3 = []
        self.pesticide_levels = []

    def progress(self, df_weather_day, pesticide_levels: np.ndarray[1, np.float32]):
        temperatures = df_weather_day["temperature_2m"].to_numpy()

        day = df_weather_day.index.date[0]
        delta_incubation = compute_delta_incubation(np.mean(temperatures))
        self.incubation.append((day, delta_incubation))

        if not self.terminated():
            hours = df_weather_day.index
            hours_since_onset = ((hours - self.discharge_date).total_seconds() / 3600).to_numpy()
            self.hours.extend(hours_since_onset)

            rains = df_weather_day['precipitation'].to_numpy()
            humidities = df_weather_day['relative_humidity_2m'].to_numpy()
            deposition_rates = compute_deposition_rate(rains, self.lai)

            hours_since_rain = items_since_last_true(is_rain_event(df_weather_day)) # TODO: take past 24 hours into account

            sigmoid_s1 = compute_ds1(self.infection_temperature, hours_since_onset)
            sigmoid_s2 = compute_ds2(self.infection_temperature, hours_since_onset)
            sigmoid_s3 = compute_ds3(self.infection_temperature, hours_since_onset)

            s3 = sigmoid_s1 * sigmoid_s2 * sigmoid_s3
            s2 = sigmoid_s1 * sigmoid_s2 - s3
            s1 = sigmoid_s1 - (s2 + s3)

            self.s1_sigmoid.extend(sigmoid_s1)
            self.s2_sigmoid.extend(sigmoid_s2)
            self.s3_sigmoid.extend(sigmoid_s3)

            self.s1.extend(s1)
            self.s2.extend(s2)
            self.s3.extend(s3)

            delta_s1 = compute_derivative_ds1(self.infection_temperature, hours_since_onset)
            s1_not_deposited = delta_s1 * (1-deposition_rates)

            dm1 = compute_ds1_mor(hours_since_rain)
            dm2 = compute_ds2_mor(hours_since_rain, temperatures, humidities)
            dm3 = compute_ds3_mor(hours_since_rain, temperatures)

            dm1 = np.clip(dm1 + pesticide_levels[: len(dm1)], 0.0, 1.0)
            dm2 = np.clip(dm2 + pesticide_levels[: len(dm2)], 0.0, 1.0)
            dm3 = dm3

            total_population = self.total_population[-1] if self.total_population else 1.0
            total_mortality = dm1 * s1 + dm2 * s2 + dm3 * s3
            total_survival = total_population - np.cumsum(s1_not_deposited)
            total_survival = total_survival * np.cumprod(1 - total_mortality)
            self.total_population.extend(total_survival)
            self.pesticide_levels.extend(pesticide_levels)

            self.mor0.extend(s1_not_deposited)
            self.mor1.extend(dm1 * s1)
            self.mor2.extend(dm2 * s2)
            self.mor3.extend(dm3 * s3)

            infection_efficiency = self.get_infection_efficiency()
            self.infection_efficiency.append((day, infection_efficiency))

            delta_risk = self.compute_delta_risk()
            cumulative_risk = self.risk[-1][1] + delta_risk if self.risk else delta_risk
            self.risk.append((day, cumulative_risk))

    def get_infection_efficiency(self):
        # Rossi p305: The value of S3 at the end of the infection period is IEinf
        # (!) Here we take S3 at the end of each day to address ambiguity regarding the duration of the infection period

        result = self.total_population[-1] * self.s3[-1]
        return result

    def compute_delta_risk(self):
        result = self.get_infection_efficiency() * (self.pat_start - self.pat_previous)
        return result

    def terminated(self):
        return bool(self.hours) and self.hours[-1] > (self.infection_duration) and self.s3_sigmoid[-1] > 0.0


def get_values_last_discharge(discharges: list[Discharge]) -> (pd.Timestamp, float):
    """
    Retrieves the discharge date and pat value from the last discharge event.

    Parameters:
    - discharges (list[Discharge]): A list of Discharge objects.

    Returns:
    - pd.Timestamp: The discharge date of the last discharge, or January 1, 1900, if the list is empty.
    - float: Pat value (proportion of ascospores that can become airborne) from last discharge, or 0 if the list is empty.
    """

    if discharges:
        time_previous = discharges[-1].discharge_date
        pat_previous = discharges[-1].pat_start
    else:
        time_previous = datetime.datetime(1900, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        pat_previous = 0
    return time_previous, pat_previous


def get_risk(infections: list[InfectionRate], date: datetime.date) -> np.float64:
    """
    Computes the risk score for a specific date, based on all infections active on that day.

    Parameters:
    - infections (list[InfectionRate]): A list of InfectionRate objects containing risk data.
    - date (datetime.date): The specific date for which to compute the risk.

    Returns:
    - np.float64: The total risk score for the specified date. Returns 0.0 if no infection is active on that day.
    """

    risks = []
    for infection in infections:
        for (risk_day, risk_score) in infection.risk:
            if risk_day == date:
                risks.append(risk_score.item())

    result = np.sum(risks) if len(risks) != 0 else 0.0
    return result


class PesticideApplication:
    """
    Represents a pesticide application event.

    Attributes:
        amount (float): The amount of pesticide applied.
        coverage (float): The coverage percentage (0.0 to 1.0).
        remaining_pesticide (float): The amount of pesticide remaining after wash-off.
    """

    def __init__(self, amount: float, removal_rate_per_mm_rain: float, application_datetime: datetime):
        self.amount = amount
        self.coverage = 1.0  # Initial coverage set to 100%
        self.remaining_pesticide = amount  # initially, all pesticide is remaining
        self.removal_rate_per_mm_rain = removal_rate_per_mm_rain
        self.application_datetime = application_datetime

    def update_remaining(self, rain: float):
        """
        Updates the remaining pesticide after considering wash-off due to rain.

        Args:
            rain (float): Amount of rain (in mm) that affects wash-off.
        """
        wash_off = self.remaining_pesticide * (self.removal_rate_per_mm_rain * rain)
        self.remaining_pesticide = max(0.0, self.remaining_pesticide - wash_off)

    def update_coverage(self, dilution_rate_per_hour=0.006):
        """
        Updates the coverage based on the dilution rate.
        """
        self.coverage = max(0.0, self.coverage - dilution_rate_per_hour)

    def is_application_valid(self, current_datetime: datetime):
        """
        Checks if the application date has passed.

        Args:
            current_datetime (datetime): The current date and time to check against.

        Returns:
            bool: True if the application date has passed, False otherwise.
        """
        return self.application_datetime <= current_datetime


class Pesticide:
    """
    Represents the amount of pesticide present over time, considering decay and application.

    Attributes:
        pesticide_levels (list[float]): A list containing the pesticide amount for each hour (24 elements).
        applications (list[PesticideApplication]): List of pesticide applications.
    """

    def __init__(self, dilution_rate_per_hour: float = 0.006):
        """
        Initializes the Pesticide class with an initial amount.

        Args:
            initial_amount (float): The initial amount of pesticide present.
        """
        self.applications = []  # List to hold all pesticide applications
        self.dilution_rate_per_hour = dilution_rate_per_hour  # Coverage diminishes by this rate per hour
        self.effective_coverage = [0.0] * 24  # Initialize effective coverage for 24 hours

    def apply_pesticide(self, amount: float, application_datetime: datetime, removal_rate_per_mm_rain: float = 0.1):
        """Applies pesticide at a specific hour."""
        self.applications.append(PesticideApplication(amount=amount, application_datetime=application_datetime,
                                                      removal_rate_per_mm_rain=removal_rate_per_mm_rain))

    def update(self, df_weather_day: pd.DataFrame, action):
        """
        Updates the pesticide levels based on rainfall and actions (application).

        Parameters:
        - df_weather_day (pd.DataFrame): A DataFrame containing hourly weather data of a selected day

        """

        rains = df_weather_day["precipitation"].to_numpy()
        rains = np.pad(rains, (0, max(0, 24 - len(rains))), "edge")
        # Reset effective coverage for each hour
        self.effective_coverage = [0.0] * 24
        day = df_weather_day.index.date[0]

        if action > 0.0:
            application_time = datetime.datetime.combine(day, datetime.datetime.min.time()) + datetime.timedelta(hours=8)
            self.apply_pesticide(amount=action, removal_rate_per_mm_rain=0.02, application_datetime=application_time)

        for hour in range(24):
            current_datetime = datetime.datetime.combine(day, datetime.datetime.min.time()) + datetime.timedelta(hours=hour)
            for application in self.applications:
                if application.is_application_valid(current_datetime):
                    application.update_remaining(rains[hour])
                    application.update_coverage(dilution_rate_per_hour=self.dilution_rate_per_hour)  # Update coverage for each application

                    # Calculate the effective coverage for the hour
                    effective_coverage_for_hour = application.remaining_pesticide * application.coverage
                    self.effective_coverage[hour] += effective_coverage_for_hour
