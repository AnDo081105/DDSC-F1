# When enter a race session and a year, 
# the code will collect data from n previous year race in the same event.

import os
import fastf1 as ff1
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

ff1.Cache.enable_cache(cache_dir)

def get_event_data(year, event_name):
    '''
    Get qualifying and race data for current year.

    Arguments:
    year -- int, the year of the event
    event_name -- str, the name of the event

    Returns:
    pd.DataFrame -- DataFrame with qualifying times and race lap times from current year
    '''

    # Get free practice data 
    fps = ['FP1', 'FP2', 'FP3']
    fp_results = {}
    for fp in fps:
        try:
            session_fp = ff1.get_session(year, event_name, fp)
            session_fp.load()

            drivers = session_fp.laps['Driver'].unique()
            for driver in drivers:
                if driver not in fp_results:
                    fp_results[driver] = {
                        'driver': driver,
                        'team': session_fp.laps.loc[session_fp.laps['Driver'] == driver, 'Team'].iloc[0],
                    }
                driver_laps = session_fp.laps.pick_driver(driver)

                if not driver_laps.empty:
                    sector_times = driver_laps[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean().dropna()             
                    total_times = sector_times.sum().total_seconds() if not sector_times.isna().any() else None

                    fp_results[driver] = {
                        **fp_results[driver],
                        f'{fp}_total_time': total_times
                    }
        except Exception as e:
            print(f"Could not load {fp} for {event_name} {year}: {e}")
            continue
    fp_df = pd.DataFrame.from_dict(fp_results, orient='index')

    # Get current year qualifying data
    try:
        session_q = ff1.get_session(year, event_name, 'Q')
        session_q.load()
        
        qualifying_laps = session_q.laps[['Driver', 'LapTime']].copy()
        qualifying_laps = qualifying_laps.dropna(subset=['LapTime'])
        
        # Get best qualifying lap per driver
        qualifying_best = qualifying_laps.groupby('Driver')['LapTime'].min().reset_index()
        qualifying_best['Qualifying_Time'] = qualifying_best['LapTime'].dt.total_seconds()
        qualifying_best = qualifying_best[['Driver', 'Qualifying_Time']]
        
    except Exception as e:
        print(f"Could not load qualifying for {event_name} {year}: {e}")
        return pd.DataFrame()
    
    # Get current year race data
    try:
        session_r = ff1.get_session(year, event_name, 'R')
        session_r.load()
        
        race_laps = session_r.laps[['Driver', 'LapTime']].copy()
        race_laps = race_laps.dropna(subset=['LapTime'])
        race_laps['Race_Time'] = race_laps['LapTime'].dt.total_seconds()
        race_laps = race_laps[['Driver', 'Race_Time']]

        # Add weather condition
        weather = session_r.weather_data[['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'Rainfall', 'Pressure']]
        if weather is not None and not weather.empty:
            for col in weather.columns:
                race_laps[f'Weather_{col}'] = weather[col].mean()
        
    except Exception as e:
        print(f"Could not load race for {event_name} {year}: {e}")
        return pd.DataFrame()
    
    # Merge qualifying and race data
    merged_df = qualifying_best.merge(race_laps, on='Driver', how='inner')
    merged_df = merged_df.merge(fp_df, left_on='Driver', right_on='driver', how='left')
    merged_df.drop(columns=['driver'], inplace=True)
    merged_df['Year'] = year
    merged_df['EventName'] = event_name
    
    return merged_df

def main():
    # Get 2026 Schedule
    schedule_2025 = ff1.get_event_schedule(2025)

    # Collect data from previous years for each event in 2025
    for _, event in schedule_2025.iterrows():
        all_data = []
        event_name = event['EventName']
        print(f"Collecting data for event: {event_name}")

        current_year = 2025
        year_collected = []
        min_year = 3
        max_year = 10

        while len(year_collected) < min_year and current_year >= 2025 - max_year:
            try:
                event_data = get_event_data(current_year, event_name)
                all_data.append(event_data)
                year_collected.append(current_year)

            except Exception as e:
                print(f"Could not collect data for {event_name} in {current_year}: {e}")
                continue
            current_year -= 1

        if all_data:
            merged_data = pd.concat(all_data, ignore_index=True)
            # Save to CSV
            dataset_dir = 'dataset'
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            csv_filename = f"{event_name.replace(' ', '_')}_data.csv"
            output_file = os.path.join(dataset_dir, csv_filename)
            merged_data.to_csv(output_file, index=False)
            print(f"Data for {event_name} saved to {output_file}")

if __name__ == "__main__":
    main()