from torch import nn
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


def compute_deposition_rate(rain, lai=1.0, height=0.5):
    ds_wet = compute_sdl_wet(rain, lai, height)
    ds_dry = compute_sdl_dry(lai)
    ds_sum = np.clip(ds_wet + ds_dry, None, 1.0)
    return ds_sum


def compute_ds2(temperature, hour_since_onset):
    result_below_20 = 1.0 / (1.0 + np.exp((5.23 - 0.1226 * temperature + 0.0014 * (temperature ** 2)) - (
            0.093 + 0.0112 * temperature - 0.000122 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.97 + 0.4297 * temperature - 0.0061 * (temperature ** 2)) - (
            0.416 - 0.0031 * temperature - 0.000245 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_derivative_ds2(temperature, hour_since_onset):
    # For result_below_20
    coef1_below_20 = 5.23 - 0.1226 * temperature + 0.0014 * (temperature ** 2)
    coef2_below_20 = 0.093 + 0.0112 * temperature - 0.000122 * (temperature ** 2)
    derivative_below_20 = -1.0 * coef2_below_20
    result_below_20 = 1.0 / (1.0 + np.exp(coef1_below_20 - coef2_below_20 * hour_since_onset))

    # For result_above_20
    coef1_above_20 = -2.97 + 0.4297 * temperature - 0.0061 * (temperature ** 2)
    coef2_above_20 = 0.416 - 0.0031 * temperature - 0.000245 * (temperature ** 2)
    derivative_above_20 = -1.0 * coef2_above_20
    result_above_20 = 1.0 / (1.0 + np.exp(coef1_above_20 - coef2_above_20 * hour_since_onset))

    # Choose the result based on the condition
    condition = np.mean(temperature) <= 20.0
    result = np.where(condition, result_below_20, result_above_20)

    # Choose the derivative based on the condition
    derivative = np.where(condition, derivative_below_20, derivative_above_20)
    # Multiply by the derivative of the function 1/(1+exp(u))
    return derivative * result * (result - 1.0)


def compute_ds3(temperature, hour_since_onset):
    result_below_20 = 1.0 / (1.0 + np.exp((6.33 - 0.0647 * temperature - 0.000317 * (temperature ** 2)) - (
            0.111 + 0.01240 * temperature - 0.000181 * (temperature ** 2)) * hour_since_onset))
    result_above_20 = 1.0 / (1.0 + np.exp((-2.13 + 0.5302 * temperature - 0.009130 * (temperature ** 2)) - (
            0.405 + 0.00079 * temperature - 0.000347 * (temperature ** 2)) * hour_since_onset))
    result = result_below_20 if np.mean(temperature) <= 20.0 else result_above_20
    return result


def compute_derivative_ds3(temperature, hour_since_onset):
    # For result_below_20
    coef1_below_20 = 6.33 - 0.0647 * temperature - 0.000317 * (temperature ** 2)
    coef2_below_20 = 0.111 + 0.01240 * temperature - 0.000181 * (temperature ** 2)
    derivative_below_20 = -1.0 * coef2_below_20
    result_below_20 = 1.0 / (1.0 + np.exp(coef1_below_20 - coef2_below_20 * hour_since_onset))

    # For result_above_20
    coef1_above_20 = -2.13 + 0.5302 * temperature - 0.009130 * (temperature ** 2)
    coef2_above_20 = 0.405 + 0.00079 * temperature - 0.000347 * (temperature ** 2)
    derivative_above_20 = -1.0 * coef2_above_20
    result_above_20 = 1.0 / (1.0 + np.exp(coef1_above_20 - coef2_above_20 * hour_since_onset))

    # Choose the result based on the condition
    condition = np.mean(temperature) <= 20.0
    result = np.where(condition, result_below_20, result_above_20)

    # Choose the derivative based on the condition
    derivative = np.where(condition, derivative_below_20, derivative_above_20)

    # Multiply by the derivative of the function 1/(1+exp(u))
    return derivative * result * (result - 1.0)


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


def compute_risk(cumulative_discharge, cumulative_ds3, host_susceptibility):
    result = cumulative_discharge * cumulative_ds3 * host_susceptibility
    return result


def compute_leaf_development(lai):
    result = 1 / (-5445.5 * (lai ** 2) + 661.55 * (lai))  # TODO: looks suspicious
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


def get_values_last_infections(infections: list):
    if infections:
        time_previous = infections[-1].discharge_date
        pat_previous = infections[-1].pat_start
    else:
        time_previous = datetime.datetime(1900, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        pat_previous = 0
    return time_previous, pat_previous


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


class InfectionRate(nn.Module):
    def __init__(self, discharge_date, ascospore_value, previous_ascospore_value, lai, duration, temperature):
        super(InfectionRate, self).__init__()
        self.discharge_date = discharge_date
        self.pat_start = ascospore_value
        self.pat_previous = previous_ascospore_value
        self.lai = lai
        self.infection_duration = duration
        self.infection_temperature = temperature
        #self.infection_duration, self.infection_temperature, self.wet = \
        #    compute_duration_and_temperature_wet_period(df_weather_infection)
        #will_infect = meets_infection_requirement(self.infection_temperature, self.infection_duration)

        self.done = False

        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0

        self.sigmoid_s1 = 0.0
        self.sigmoid_s2 = 0.0
        self.sigmoid_s3 = 0.0

        self.s1_rate = 0.0
        self.s2_rate = 0.0
        self.s3_rate = 0.0

        self.hours_progress = []

        self.s1_sigmoid_progress = []
        self.s2_sigmoid_progress = []
        self.s3_sigmoid_progress = []

        self.s0_progress = []
        self.s1_progress = []
        self.s2_progress = []
        self.s3_progress = []
        self.total_population_progress = []

        self.s1_rate_progress = []
        self.s2_rate_progress = []
        self.s3_rate_progress = []

        self.mor1_progress = []
        self.mor2_progress = []
        self.mor3_progress = []

        self.has_reached_s1 = 0.0
        self.has_reached_s2 = 0.0
        self.has_reached_s3 = 0.0

        self.has_reached_s1_progress = []
        self.has_reached_s2_progress = []
        self.has_reached_s3_progress = []

        self.dep = []
        self.ger = []
        self.app = []

        self.s2_fake = 1.0
        self.s2_fake_progress = []

    def progress(self, df_weather_day):
        if self.done: return
        hours = df_weather_day.index
        hours_since_onset = ((hours - self.discharge_date).total_seconds() / 3600).to_numpy()
        self.hours_progress.extend(hours_since_onset)
        if hours_since_onset[-1] > self.infection_duration:
            self.done = True
        temperatures = df_weather_day['temperature_2m'].to_numpy()
        rains = df_weather_day['precipitation'].to_numpy()
        humidities = df_weather_day['relative_humidity_2m'].to_numpy()
        deposition_rates = compute_deposition_rate(rains, self.lai.value).numpy()
        hours_since_rain = items_since_last_true(
            is_rain_event(df_weather_day))  # issue: past 24 hours not taken into account

        sigmoid_s0 = 1.0 - compute_ds1(self.infection_temperature, hours_since_onset)
        sigmoid_s1 = compute_ds1(self.infection_temperature, hours_since_onset)
        sigmoid_s2 = compute_ds2(self.infection_temperature, hours_since_onset)
        sigmoid_s3 = compute_ds3(self.infection_temperature, hours_since_onset)

        s3 = sigmoid_s1 * sigmoid_s2 * sigmoid_s3
        s2 = sigmoid_s1 * sigmoid_s2 - s3
        s1 = sigmoid_s1 - (s2 + s3)
        s0 = sigmoid_s0

        #self.s0_progress.append(s0)
        #self.s1_progress.append(s1)
        #self.s2_progress.append(s2)
        #self.s3_progress.append(s3)

        dm1 = compute_ds1_mor(hours_since_rain)
        dm2 = compute_ds2_mor(hours_since_rain, temperatures, humidities)
        dm3 = compute_ds3_mor(hours_since_rain, temperatures)

        total_population = self.total_population_progress[-1] if self.total_population_progress else 1.0
        total_mortality = dm1 * s1 + dm2 * s2 + dm3 * s3
        total_survival = total_population * np.cumprod(1 - total_mortality)
        self.total_population_progress.extend(total_survival)

        self.mor1_progress.extend(dm1 * s1)
        self.mor2_progress.extend(dm2 * s2)
        self.mor3_progress.extend(dm3 * s3)

        for temperature, hour_since_onset, rain, humidity, deposition_rate, hour_since_rain in \
                zip(temperatures, hours_since_onset, rains, humidities, deposition_rates, hours_since_rain):
            deposition_rate = 1.0  # TODO: remove

            # for each step compute:
            # fraction_{ds1,ds2,ds3} under no_mortality assumption
            # compute mortality for {ds1,ds2,ds3}
            # posthoc subtract died_population

            dm1 = compute_ds1_mor(hour_since_rain)
            dm2 = compute_ds2_mor(hour_since_rain, temperature, humidity)
            dm3 = compute_ds3_mor(hour_since_rain, temperature)

            if True:  # TODO: remove
                dm1, dm2, dm3 = 0, 0, 0

            mor1 = dm1 * self.s1
            mor2 = dm2 * self.s2
            mor3 = dm3 * self.s3

            self.sigmoid_s1 = compute_ds1(self.infection_temperature, hour_since_onset)
            ds1 = compute_derivative_ds1(self.infection_temperature, hour_since_onset)
            dep = ds1 * deposition_rate
            self.has_reached_s1 = self.has_reached_s1 + dep

            self.sigmoid_s2 = compute_ds2(self.infection_temperature, hour_since_onset)
            ds2 = compute_derivative_ds2(self.infection_temperature, hour_since_onset)
            ger = ds2 * self.sigmoid_s1 + self.sigmoid_s2 * ds1  # product rule
            self.has_reached_s2 = self.has_reached_s2 + ger

            self.sigmoid_s3 = compute_ds3(self.infection_temperature, hour_since_onset)
            ds3 = compute_derivative_ds3(self.infection_temperature, hour_since_onset)
            app = ds3 * self.sigmoid_s1 * self.sigmoid_s2 + self.sigmoid_s3 * ger
            self.has_reached_s3 = self.has_reached_s3 + app

            self.s1_rate = dep - ger
            self.s2_rate = ger - app
            self.s3_rate = app

            self.dep.append(dep)
            self.ger.append(ger)
            self.app.append(app)

            self.s1 = self.s1 + self.s1_rate
            self.s2 = self.s2 + self.s2_rate
            self.s3 = self.s3 + self.s3_rate

            self.s2_fake = self.s2_fake - app
            self.s2_fake_progress.append(self.s2_fake)

            self.s1_progress.append(self.s1)
            self.s2_progress.append(self.s2)
            self.s3_progress.append(self.s3)

            self.s1_rate_progress.append(self.s1_rate)
            self.s2_rate_progress.append(self.s2_rate)
            self.s3_rate_progress.append(self.s3_rate)

            self.s1_rate = 0.0
            self.s2_rate = 0.0
            self.s3_rate = 0.0

            self.s1_sigmoid_progress.append(self.sigmoid_s1)
            self.s2_sigmoid_progress.append(self.sigmoid_s2)
            self.s3_sigmoid_progress.append(self.sigmoid_s3)

            self.has_reached_s1_progress.append(self.has_reached_s1)
            self.has_reached_s2_progress.append(self.has_reached_s2)
            self.has_reached_s3_progress.append(self.has_reached_s3)
