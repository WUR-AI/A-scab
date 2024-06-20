from torch import nn
import numpy as np

from ascab.utils.weather import is_rain_event
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


def compute_derivative_ds1_numerical(temperature, hour_since_onset, timestep=1.0):
    delta_y = compute_ds1(temperature, hour_since_onset) - compute_ds1(temperature, hour_since_onset - timestep)
    delta_x = timestep
    return delta_y / delta_x


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


def compute_ds2_mor():
    return 0  # TODO implement eq 21


def compute_ds3_mor():
    return 0  # TODO implement eq 22


def compute_leaf_development(lai):
    result = 1 / (-5445.5 * lai ** 2 + 661.55 * lai)
    return result


def get_discharge_date(df_weather_day, pat_previous, pat_current, time_previous):
    # issue: past 24 hours not taken into account
    rain_events = is_rain_event(df_weather_day)
    is_daytime = df_weather_day['is_day'].to_numpy()
    hours_since_previous = (df_weather_day.index - time_previous).total_seconds() / 3600
    discharge_hour_index = determine_discharge_hour_index(pat_current, rain_events, is_daytime, pat_previous,
                                                          hours_since_previous)
    if discharge_hour_index is None: return None
    discharge_date = df_weather_day.index[discharge_hour_index]
    return discharge_date


class InfectionRate(nn.Module):
    def __init__(self, discharge_date, ascospore_value, lai):
        super(InfectionRate, self).__init__()
        self.discharge_date = discharge_date
        self.pat_start = ascospore_value
        self.lai = lai

        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0

        self.sigmoid_s1 = 0.0
        self.sigmoid_s2 = 0.0
        self.sigmoid_s3 = 0.0

        self.s1_rate = 0.0
        self.s2_rate = 0.0
        self.s3_rate = 0.0

        self.s1_sigmoid_progress = []
        self.s2_sigmoid_progress = []
        self.s3_sigmoid_progress = []

        self.s1_progress = []
        self.s2_progress = []
        self.s3_progress = []

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
        hours = df_weather_day.index
        hours_since_onset = ((hours - self.discharge_date).total_seconds() / 3600).to_numpy()
        temperatures = df_weather_day['temperature_2m'].to_numpy()
        rains = df_weather_day['precipitation'].to_numpy()
        deposition_rates = compute_deposition_rate(rains, self.lai.value).numpy()
        hours_since_rain = items_since_last_true(
            is_rain_event(df_weather_day))  # issue: past 24 hours not taken into account
        for _, hour_since_onset, rain, deposition_rate, hour_since_rain in zip(temperatures, hours_since_onset, rains,
                                                                               deposition_rates, hours_since_rain):
            temperature = 20.0  # TODO: remove
            deposition_rate = 1.0  # TODO: remove

            dm1 = 0  # compute_ds1_mor(hour_since_rain)
            dm2 = compute_ds2_mor()
            dm3 = compute_ds3_mor()

            mor1 = dm1 * self.s1
            mor2 = dm2 * self.s2
            mor3 = dm3 * self.s3

            self.sigmoid_s1 = compute_ds1(temperature, hour_since_onset)
            ds1 = compute_derivative_ds1(temperature, hour_since_onset)
            dep = ds1 * deposition_rate
            self.has_reached_s1 = self.has_reached_s1 + dep

            self.sigmoid_s2 = compute_ds2(temperature, hour_since_onset)
            ds2 = compute_derivative_ds2(temperature, hour_since_onset)
            ger = ds2 * self.sigmoid_s1 + self.sigmoid_s2 * ds1  # product rule
            self.has_reached_s2 = self.has_reached_s2 + ger

            self.sigmoid_s3 = compute_ds3(temperature, hour_since_onset)
            ds3 = compute_derivative_ds3(temperature, hour_since_onset)
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

            self.mor1_progress.append(dm1)
            self.mor2_progress.append(dm2)
            self.mor3_progress.append(dm3)

            self.has_reached_s1_progress.append(self.has_reached_s1)
            self.has_reached_s2_progress.append(self.has_reached_s2)
            self.has_reached_s3_progress.append(self.has_reached_s3)