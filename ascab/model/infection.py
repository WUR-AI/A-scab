import numpy as np
import pytz
import datetime

from ascab.utils.weather import is_rain_event, compute_duration_and_temperature_wet_period
from ascab.utils.generic import items_since_last_true


def determine_discharge_hour_index(pat, rain_events, is_daytime, pat_previous, hours_since_previous):
    # no discharge
    if np.logical_or(pat <= 0.016, 1 not in rain_events):
        return None

    first_rain_hour = np.where(rain_events == 1)[0][0]
    heavy_dew = False
    # Rossi et al. page 304 (top left)
    if np.logical_or(np.logical_or(hours_since_previous[first_rain_hour] < 5.0, heavy_dew),
                     np.logical_and(first_rain_hour <= 7, pat <= 0.80)):
        return None

    # no delay
    if np.logical_or(np.logical_or(pat >= 0.80, (pat - pat_previous) >= 0.30), is_daytime[first_rain_hour]):
        return first_rain_hour

    # delay: sunset
    first_daytime = np.where(is_daytime == 1)[0][0]
    return first_daytime


def compute_ds1(temperature, hour_since_onset):
    result = np.array(1.0 / (1.0 + np.exp(2.999 - 0.067 * temperature * hour_since_onset)))
    mask = hour_since_onset < 0
    result[mask] = 0
    return result


def compute_derivative_ds1(temperature, hour_since_onset):
    u = 2.999 - 0.067 * temperature * hour_since_onset
    result = 0.067 * temperature * np.exp(u) / ((1.0 + np.exp(u)) ** 2)
    # result = np.where(hour_since_onset >= 0, result, 0.0)
    return result


def compute_sdl_wet(rain, lai=1.0, height=0.5):
    lambda_h = (1.0 / (1.0 + np.exp(2.575 - 0.987 * lai * (5.022 * (rain ** 0.063)))))
    result = (1.017 * 0.374 ** height) * lambda_h
    return result


def compute_sdl_dry(lai=1.0):
    result = 0.594 - (0.643 * 0.372 ** lai)
    return result


def compute_deposition_rate(rain, lai=1.0, height=0.5, do_clip = True):
    ds_wet = compute_sdl_wet(rain, lai, height)
    ds_dry = compute_sdl_dry(lai)
    ds_sum = ds_wet + ds_dry
    if do_clip:
        ds_sum = np.clip(ds_sum, None, 1.0)
    return ds_sum


def compute_ds2(temperature, hour_since_onset):
    result_below_20 = 1.0 / (1.0 + np.exp((5.23 - 0.1226 * temperature + 0.0014 * (temperature ** 2)) - (
            0.093 + 0.0112 * temperature - 0.000122 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.97 + 0.4297 * temperature - 0.0061 * (temperature ** 2)) - (
            0.416 - 0.0031 * temperature - 0.000245 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_ds3(temperature, hour_since_onset):
    result_below_20 = 1.0 / (1.0 + np.exp((6.33 - 0.0647 * temperature - 0.000317 * (temperature ** 2)) - (
            0.111 + 0.01240 * temperature - 0.000181 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.13 + 0.5302 * temperature - 0.009130 * (temperature ** 2)) - (
            0.405 + 0.00079 * temperature - 0.000347 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_ds1_mor(hour_since_last_rain):
    result = 0.263 * (1 - 0.97315 ** hour_since_last_rain)
    return result


def compute_ds2_mor(hour_since_last_rain, temperature, humidity):
    result = (-1.538 + 0.253 * temperature - 0.00694 * (temperature ** 2)) * \
             (1 - 0.977 ** hour_since_last_rain) * (0.0108 * humidity - 0.008)
    return result


def compute_ds3_mor(hour_since_last_rain, temperature):
    result = (0.0028 * hour_since_last_rain) * \
             (-1.27 + 0.326 * temperature - 0.0102 * (temperature ** 2))
    return result


def compute_leaf_development(lai):
    result = 1 / (-5445.5 * (lai ** 2) + 661.55 * (lai))  # TODO: looks suspicious
    return result


def compute_delta_incubation(temperature):
    result = 1.0 / (26.4 - 1.0268 * temperature)
    return result


def get_discharge_date(df_weather_day, pat_previous, pat_current, time_previous):
    if pat_current > 0.99: return None
    # issue: past 24 hours not taken into account
    rain_events = is_rain_event(df_weather_day)
    is_daytime = df_weather_day['is_day'].to_numpy()
    hours_since_previous = (df_weather_day.index - time_previous).total_seconds() / 3600
    discharge_hour_index = determine_discharge_hour_index(pat_current, rain_events, is_daytime, pat_previous,
                                                          hours_since_previous)
    if discharge_hour_index is None: return None
    discharge_date = df_weather_day.index[discharge_hour_index]
    return discharge_date


def meets_infection_requirement(temperature, wet_hours):
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


def will_infect(df_weather_infection):
    infection_duration, infection_temperature = compute_duration_and_temperature_wet_period(df_weather_infection)
    result = meets_infection_requirement(infection_temperature, infection_duration)
    return result, infection_duration, infection_temperature


class InfectionRate():
    def __init__(self, discharge_date, ascospore_value, previous_ascospore_value, lai, duration, temperature):
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

    def progress(self, df_weather_day, action=0):
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

            if action:
                dm1[:] = action
                dm2[:] = action
                dm3[:] = action

            total_population = self.total_population[-1] if self.total_population else 1.0
            total_mortality = dm1 * s1 + dm2 * s2 + dm3 * s3
            total_survival = total_population - np.cumsum(s1_not_deposited)
            total_survival = total_survival * np.cumprod(1 - total_mortality)
            self.total_population.extend(total_survival)

            self.mor0.extend(s1_not_deposited)
            self.mor1.extend(dm1 * s1)
            self.mor2.extend(dm2 * s2)
            self.mor3.extend(dm3 * s3)

            delta_infection_efficiency = self.get_infection_efficiency()
            self.infection_efficiency.append((day, delta_infection_efficiency))

            delta_risk = self.compute_delta_risk()
            cumulative_risk = self.risk[-1][1] + delta_risk if self.risk else delta_risk
            self.risk.append((day, cumulative_risk))

    def get_infection_efficiency(self):
        result = self.total_population[-1] * self.s3[-1]
        return result

    def compute_delta_risk(self):
        result = self.get_infection_efficiency() * (self.pat_start - self.pat_previous)
        return result

    def terminated(self):
        return bool(self.hours) and self.hours[-1] > self.infection_duration


def get_values_last_infections(infections: list[InfectionRate]):
    if infections:
        time_previous = infections[-1].discharge_date
        pat_previous = infections[-1].pat_start
    else:
        time_previous = datetime.datetime(1900, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        pat_previous = 0
    return time_previous, pat_previous


def get_risk(infections: list[InfectionRate], date):
    risks = []
    for infection in infections:
        for (risk_day, risk_score) in infection.risk:
            if risk_day == date:
                risks.append(risk_score.item())

    result = np.sum(risks) if len(risks) != 0 else 0
    return result
