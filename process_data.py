import datetime
import math
import os
import urllib

import numpy as np
import pandas as pd
import streamlit as st


def download_data(wait_hours=6):
    """
    Periodically download data to csv
    """

    # Download new data when last mod time was > x hours
    filepath = "vaccine.csv"  # 'daily.csv'
    last_mod = os.path.getmtime(filepath)
    last_mod = datetime.datetime.utcfromtimestamp(last_mod)
    dif = datetime.datetime.now() - last_mod
    if dif < datetime.timedelta(hours=wait_hours) and os.path.exists(
        "Region_Mobility_Report_CSVs"
    ):
        return

    # Clear cache if we have new data
    st.caching.clear_cache()

    with st.spinner("Fetching latest data..."):
        os.remove("daily.csv")
        os.remove("states_daily.csv")
        urllib.request.urlretrieve(
            "https://api.covidtracking.com/v1/us/daily.csv", "daily.csv"
        )
        urllib.request.urlretrieve(
            "https://api.covidtracking.com/v1/states/daily.csv", "states_daily.csv"
        )


@st.cache(suppress_st_warning=True)
def process_data(all_states, state):
    """
    Process CSVs. Smooth and compute new series.

    :param all_states: Boolean if "all states" is checked
    :param state: Selected US state
    :return: Dataframe
    """
    # Data
    if all_states:
        df = pd.read_csv("daily.csv").sort_values("date", ascending=True).reset_index()
    else:
        df = (
            pd.read_csv("states_daily.csv")
            .sort_values("date", ascending=True)
            .reset_index()
            .query('state=="{}"'.format(state))
        )

    df = df.query("date >= 20200301")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.set_index("date", inplace=True)

    # Rolling means
    df["positiveIncrease"] = df["positiveIncrease"].rolling(7).mean()
    df["deathIncrease"] = df["deathIncrease"].rolling(7).mean()
    df["hospitalizedCurrently"] = df["hospitalizedCurrently"].rolling(7).mean()
    df["totalTestResultsIncrease"] = df["totalTestResultsIncrease"].rolling(7).mean()

    # New features
    df["percentPositive"] = (
        (df["positiveIncrease"] / df["totalTestResultsIncrease"]).rolling(7).mean()
    )
    df["Case Fatality Rate"] = (df["death"] / df["positive"]) * 100

    df = calc_prevalence_ratio(df)

    df["Infection Fatality Rate"] = (
        df["death"] / (df["positive"] * df["prevalence_ratio"])
    ) * 100
    df["percentPositive"] = df["percentPositive"] * 100
    df["Cumulative Recovered Infections Estimate"] = (
        df["positive"] * df["prevalence_ratio"] - df["death"]
    )

    if np.inf in df.values:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def calc_prevalence_ratio(df):
    """
    Calculate prevalence ratio
    prevalence_ratio(day_i) = (1250 / (day_i + 25)) * (positivity_rate(day_i))^(0.5) + 2, where day_i is the number of days since February 12, 2020.
    https://covid19-projections.com/estimating-true-infections-revisited/

    :param df: Dataframe from process_data()
    :return: Dataframe with prevalence_ratio column
    """

    days_since = df.index - datetime.datetime(year=2020, month=2, day=12)
    df["days_since_feb12"] = days_since.days.values
    p_r_list = []
    for i, row in df.iterrows():
        try:
            prevalence_ratio = (1000 / (row["days_since_feb12"] + 10)) * math.pow(
                row["percentPositive"], 0.5
            ) + 2
        except:
            prevalence_ratio = p_r_list[-1]
        p_r_list.append(prevalence_ratio)
    df["prevalence_ratio"] = p_r_list
    return df
