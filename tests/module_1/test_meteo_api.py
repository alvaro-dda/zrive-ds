""" This is a dummy example to show how to import code from src/ for testing"""

import pandas as pd
from src.module_1.module_1_meteo_api import call_API, get_data_meteo_api, transform_data


def test_call_API():
    test = call_API(
        "https://climate-api.open-meteo.com/v1/climate?latitude=52.52&longitude=13.41&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily=temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean" # noqa
    )
    assert "daily" in test.keys()


def test_get_data_meteo_api():
    assert "daily" in get_data_meteo_api("Madrid").keys()


def test_transform_data():
    json_file = get_data_meteo_api("Madrid")
    df = pd.DataFrame(json_file["daily"])
    df_soilmoist = df.iloc[:, 0::3]
    assert "Average" in transform_data(df_soilmoist)
    assert "Dispersion" in transform_data(df_soilmoist)
    assert "year" in transform_data(df_soilmoist)
