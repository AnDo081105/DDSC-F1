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
    Get each session data for a given event and year, including weather conditions.

    Arguments:
    year -- int, the year of the event
    event_name -- str, the name of the event

    Returns:
    pd.DataFrame -- DataFrame containing driver data and weather conditions for each session
    '''
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    final_results = {}

    for session_type in session_types:
        try:
            session = ff1.get_session(year, event_name, session_type)
            session.load()

            # Add weather condition for each session
            weather = session.weather_data
            weather_condition = {}
            if weather is not None and not weather.empty:
                weather_condition = {
                    f'{session_type}_air_temp': weather['AirTemp'].mean(),
                    f'{session_type}_humidity': weather['Humidity'].mean(),
                    f'{session_type}_rainfall': weather['Rainfall'].mean(),
                    f'{session_type}_pressure': weather['Pressure'].mean(),
                    f'{session_type}_wind_speed': weather['WindSpeed'].mean(),
                    f'{session_type}_wind_direction': weather['WindDirection'].mean(),
                    f'{session_type}_track_temp': weather['TrackTemp'].mean()
                }

            # Choose unique drivers in the session
            drivers = session.laps['Driver'].unique()
            for driver in drivers:

                # Initialize driver entry if not already present
                if driver not in final_results:
                    
                    # Basic driver info
                    final_results[driver] = {
                        'driver_number': session.laps.loc[session.laps['Driver'] == driver, 'DriverNumber'].iloc[0],
                        'driver': driver,
                        'year': year,
                        'event': event_name,
                        'team': session.laps.loc[session.laps['Driver'] == driver, 'Team'].iloc[0],
                    }
                driver_laps = session.laps.pick_driver(driver)
                
                # For each driver laps, get average sector times and total time
                if not driver_laps.empty:
                    sector_times = driver_laps[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean().dropna()             
                    total_times = sector_times.sum().total_seconds() if not sector_times.isna().any() else None
                    final_results[driver] = {
                        **final_results[driver],
                        **weather_condition,
                        f'{session_type}_total_time': total_times
                    }
            
            # In race, get final classified position
            if session_type == 'R':
                result_positions = session.results[['ClassifiedPosition', 'Abbreviation']]
                result_positions_df = pd.DataFrame(result_positions).set_index('Abbreviation')
                for driver in drivers:
                    if driver in result_positions_df.index:
                        final_results[driver]['race_position'] = result_positions_df.loc[driver, 'ClassifiedPosition']
   
        except Exception as e:
            print(f"Could not load session {session_type} for {event_name} {year}: {e}")
            continue

    df = pd.DataFrame(final_results).T

    return df

def main():
    arg_parser = argparse.ArgumentParser(description="F1 Data Pipeline")
    arg_parser.add_argument('--event', type=str, required=True, help='Event name (e.g., "Australia Grand Prix")')
    arg_parser.add_argument('--year', type=int, required=True, help='Year of the event (e.g., 2025)')
    arg_parser.add_argument('--min_year', type=int, default=3, help='Minimum number of previous years of data to collect')
    arg_parser.add_argument('--max_year', type=int, default=10, help='Maximum number of previous years of data to collect')
    args = arg_parser.parse_args()

    event_name = args.event
    year = args.year
    min_year = args.min_year
    max_year = args.max_year

    all_data = []
    years_collected = []
    current_year = year 

    # Keep goin back till enough years of data collected
    while len(years_collected) <= min_year and current_year >= year - max_year:
        try:
            print(f"Collecting data for {event_name} {current_year}")
            event_data = get_event_data(current_year, event_name)
            if not event_data.empty:
                all_data.append(event_data)
                years_collected.append(current_year)
                print(f"Data collected for {event_name} {current_year}")
        except Exception as e:
            print(f"Error collecting data for {event_name} {current_year}: {e}")
        current_year -= 1

    if len(years_collected) < min_year:
        print(f"Warning: Only collected data for {len(years_collected)} years, which is less than the minimum required {min_year} years.")
    else:
        print(f"Successfully collected data for {len(years_collected)} years.")
        
        
    # Combine all years data
    if all_data:
        df = pd.concat(all_data, ignore_index=True)

        # Dataset Folder
        dataset_dir = 'dataset'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        output_file = os.path.join(dataset_dir, f"{event_name.replace(' ', '_')}_{year}_data.csv")
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file} in dir: {dataset_dir}")

        return df
    else:
        print("No data collected.")
        return None

if __name__ == "__main__":
    main()