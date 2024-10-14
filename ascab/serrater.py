import pandas as pd
from datetime import timedelta
import math
import numpy as np

from ascab.utils.plot import plot_results
from ascab.env.env import AScabEnv, MultipleWeatherASCabEnv, get_weather_library
from ascab.train import CheatingAgent, ZeroAgent, ScheduleAgent


# Function to correct '24:00' and remove half-hour times
def correct_time(day, hour):
    if hour == "24:00":
        # Increment the day by one for '24:00'
        new_day = pd.to_datetime(day, format="%d/%m/%Y") + timedelta(days=1)
        return new_day.strftime("%d/%m/%Y") + " 00:00"
    elif ":00" in hour:
        return day + " " + hour
    else:
        return None


def vapour_pressure_deficit(temp, relative_humidity):
    saturated = 0.6108 * math.exp((17.27 * temp) / (237.3 + temp))
    vapour_pressure = relative_humidity * 0.01 * saturated
    result = saturated - vapour_pressure
    return result


def read_weather(file_path_weather):
    df_weather = pd.read_excel(file_path_weather)
    df_weather.rename(
        columns={
            "Temperatura": "temperature_2m",
            "Humitat relativa": "relative_humidity_2m",
            "Precipitació": "precipitation",
        },
        inplace=True,
    )
    df_weather["is_day"] = 1
    df_weather["vapour_pressure_deficit"] = df_weather.apply(
        lambda row: vapour_pressure_deficit(
            row["temperature_2m"], row["relative_humidity_2m"]
        ),
        axis=1,
    )
    df_weather = df_weather.rename(
        columns={df_weather.columns[0]: "DAY", df_weather.columns[1]: "HOUR"}
    )
    # Apply the correction and remove rows with half-hour times
    df_weather["date"] = df_weather.apply(
        lambda row: correct_time(row["DAY"], row["HOUR"]), axis=1
    )
    # Drop rows where DATE is None (i.e., half-hour times)
    df_weather = df_weather.dropna(subset=["date"])
    # Convert to datetime
    df_weather["date"] = pd.to_datetime(df_weather["date"], format="%d/%m/%Y %H:%M")
    df_weather["date"] = df_weather["date"].dt.tz_localize(
        "Europe/Madrid", ambiguous=False, nonexistent="shift_forward"
    )
    df_weather.set_index("date", inplace=True)
    df_weather = df_weather.sort_index()
    df_weather.replace("--", np.nan, inplace=True)

    return df_weather


def get_category_mapping():
    return {
        "NEEMAZAL (BIO 16)": ["insecticide"],
        "FERTILIMP PK ": ["insecticide", "fertilizer"],
        "EXTER SULFO SABO ": ["fungicide"],
        "SUNCLAY PROTECT CAOLIN BLANCO NATURAL": ["sunburn"],
        "CURATION": ["unknown"],
        "COURE VITRA DROXICUPER-HIDROBLUE-50 WP ": ["fungicide"],
        "COCTEL GOLD 5 LT.(GLISOFATO) ": ["herbicide"],
        "CLINIC / ROTUNDO TOP 360": ["herbicide"],
        "PUFFER CARPO/CONFUSIO CARPOC (Feromona)": ["insecticide"],
        "FEROMONA ISONET Z /CHECMATE CM F.POMONEL": ["insecticide"],
        "RONDOUP  R TRANSORB TOUCHDOWN TOMCATO": ["herbicide"],
        "SCATTO": ["insecticide"],
        "SUNDEK": ["insecticide"],
        "EMPEROR 1 LT": ["biostimulant"],
        "EMPEROR 1 LT ": ["biostimulant"],
        "CLOSER ": ["insecticide"],
        "KUMULUS - SOFRE - MICROCOPS AZUMO ": ["fungicide"],
        "VONDOZEB": ["fungicide"],
        "LUQSABIT CALÇ PROCAL": ["growth regulator"],
        "KESHET 1LT  AUDACE  DELTAPLAN CORAZA": ["insecticide"],
        "SOLUBOR": ["fertilizer"],
        "MERPAM 80 WDG (CAPTAN)": ["fungicide"],
        "DASH SC 1LT": ["adjuvant"],
        "CHORUS PREMIER": ["fungicide"],
        "ELFER AQUA (CONT PH)": ["adjuvant"],
        "GAZEL": ["insecticide"],
        "DELAN (1LT.) EFUZIN": ["fungicide"],
        "REGULEX / NOVAGIB / KEYGIB / GIBB PLUS": ["growth regulator"],
        "MOVENTO": ["insecticide"],
        "CORAGEN": ["insecticide"],
        "ATOMINAL /JUVINAL ALAZIN / EXPEDIENT": ["insecticide"],
        "TEPPEKI": ["insecticide"],
        "NIMROD": ["fungicide"],
        "REGALIS": ["growth regulator"],
        "SULFAT AMONIC /FOSFAT MONOAMONIC ": ["biostimulant"],
        "APACHE": ["insecticide"],
        "BELLIS TERRASORB RADICULAR": ["biostimulant"],
        "SYLLIT 65": ["fungicide"],
        "RHODOFIX (5 KGS.) FRUIT FIX ETIFIX": ["growth regulator"],
        "BREVIS": ["chemical thinning"],
        "EVO 2 CARPOVIRUSINE 1L": ["insecticide"],
    }


def read_treatment(file_path_treatments, spray_dict=get_category_mapping()):
    df_treatments = pd.read_excel(file_path_treatments)
    # Remove empty rows before processing
    df_treatments = df_treatments.dropna(how="all")
    # Remove rows that start with "-"
    df_treatments = df_treatments[
        ~df_treatments.apply(lambda row: row.str.startswith("-").any(), axis=1)
    ]
    # Reset index after removing rows
    df_treatments.reset_index(drop=True, inplace=True)
    first_column = df_treatments.columns[0]
    df_treatments = df_treatments.iloc[:, 1:]
    df_treatments = df_treatments.rename(
        columns={df_treatments.columns[0]: first_column}
    )
    df_treatments["Spray"] = df_treatments["Des Seccio Lin."].map(spray_dict)
    df_treatments.rename(columns={"Data": "DAY"}, inplace=True)
    return df_treatments


def get_spraying_days(file_path_treatments):
    df_treatments = read_treatment(file_path_treatments)
    df_treatments = df_treatments[df_treatments["Des. Seccio Cap."] == "CG PINK LADY 1"]
    df_fungicides = df_treatments[
        df_treatments["Spray"].apply(
            lambda x: "fungicide" in x if isinstance(x, list) else False
        )
    ]
    spraying_days = [pd.to_datetime(day).date() for day in df_fungicides["DAY"]]
    return spraying_days


if __name__ == "__main__":
    ascab = AScabEnv(
        location=(42.1620, 3.0924), dates=("2022-02-01", "2022-10-01")
    )
    ascab_cheating = cheating_agent(ascab, render=False)

    ascab_fixed_schedule = fixed_schedule_agent(ascab, dates=get_spraying_days(), render=False)
    plot_results({"AI Agent": ascab_cheating, "Standard Practice": ascab_fixed_schedule}, variables=["HasRain", "LeafWetness", "AscosporeMaturation", "Ascospores", "Discharge", "Infections", "Risk", "Action"])