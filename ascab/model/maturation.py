import datetime
import numpy as np
from ascab.utils.weather import compute_leaf_wetness_duration, is_wet


def pseudothecial_development_has_ended(stage):
    return stage >= 9.5


def pat(dhw):
    return 1.0 / (1.0 + np.exp(6.89 - 0.035 * dhw))


class PseudothecialDevelopment():
    def __init__(self, initial_value=5.0):
        super(PseudothecialDevelopment, self).__init__()
        self.value = initial_value
        self.rate = 0

    def update_rate(self, df_weather_day):
        day = df_weather_day.index.date[0].timetuple().tm_yday
        avg_temperature = df_weather_day['temperature_2m'].mean()
        total_rain = df_weather_day['precipitation'].sum()
        hours_humid = len(df_weather_day[df_weather_day['relative_humidity_2m'] > 85.0])
        wetness_duration = compute_leaf_wetness_duration(df_weather_day)
        self.rate = self.compute_rate(self.value, day, avg_temperature, total_rain, hours_humid, wetness_duration)
        return self.rate

    def compute_rate(self, current_value, day, avg_temperature, total_rain, humid_hours, wetness_duration):
        # Calculate the daily change in pseudothecial development
        dy_dt = 0.0031 + 0.0546 * avg_temperature - 0.00175 * (avg_temperature ** 2)
        # Check conditions and modify dy_dt accordingly
        start_day = datetime.datetime.strptime('February 1', '%B %d').timetuple().tm_yday
        condition = (day < start_day) or pseudothecial_development_has_ended(current_value) or (
                avg_temperature <= 0) or (total_rain <= 0.25) or (humid_hours <= 8) or (wetness_duration <= 8.0)
        dy_dt = np.where(condition, 0.0, dy_dt)
        return dy_dt

    def integrate(self):
        self.value += self.rate * 1.0


class AscosporeMaturation():
    def __init__(self, dependency):
        super(AscosporeMaturation, self).__init__()
        self.value = 0
        self.rate = 0
        self._dhw = 0
        self._delta_dhw = 0
        self._dependencies = dependency

    def update_rate(self, df_weather_day):
        precipitation = df_weather_day['precipitation'].values
        vapour_pressure_deficit = df_weather_day['vapour_pressure_deficit'].values
        temperature_2m = df_weather_day['temperature_2m'].values
        self.rate, self._delta_dhw = self.compute_rate(self._dependencies.value, self._dhw, precipitation,
                                                             vapour_pressure_deficit, temperature_2m)
        return self.rate

    def compute_rate(self, pseudothecia, current_dhw, precipitation, vapour_pressure_deficit, temperature_2m):
        if not pseudothecial_development_has_ended(pseudothecia):
            return 0, 0

        wet_hourly = is_wet(precipitation, vapour_pressure_deficit)
        hw = wet_hourly * temperature_2m / 24.0
        dhw = np.sum(hw)
        current_value = pat(current_dhw)
        new_value = pat(current_dhw + dhw)
        delta_value = new_value - current_value
        delta_dhw = dhw
        return delta_value, delta_dhw

    def integrate(self):
        self.value += self.rate * 1.0
        self._dhw += self._delta_dhw * 1.0


class LAI():
    def __init__(self):
        super(LAI, self).__init__()
        self.value = 0

    def update_rate(self, df_weather_day):
        day = df_weather_day.index.date[0].timetuple().tm_yday
        avg_temperature = df_weather_day['temperature_2m'].mean()
        self.rate = self.compute_rate(self.value, day, avg_temperature)
        return self.rate

    def compute_rate(self, current_value, day, avg_temperature):
        # Calculate the daily change
        number_of_shoots_per_m2 = 85 #50 TODO: check
        dy_dt = 0.00008 * max(0, (avg_temperature - 4.0)) * number_of_shoots_per_m2
        # Check conditions and modify dy_dt accordingly
        start_day = datetime.datetime.strptime('April 1', '%B %d').timetuple().tm_yday
        condition = (day < start_day) or (current_value > 5.0)
        dy_dt = np.where(condition, 0.0, dy_dt)
        return dy_dt

    def integrate(self):
        self.value += self.rate * 1.0
