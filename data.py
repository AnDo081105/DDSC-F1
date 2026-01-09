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
    # Get arguments from command line
    parser = argparse.ArgumentParser(description='Get F1 event data')

    parser.add_argument('--year', type=int, required=True, help='Year of the event')
    parser.add_argument('--event', type=str, required=True, help='Name of the event (e.g., "Australia Grand Prix")')

    args = parser.parse_args()

    print(f"Getting data for {args.event} {args.year}...")
    df = get_event_data(args.year, args.event)

    # Dataset Folder
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    output_file = os.path.join(dataset_dir, f"{args.event.replace(' ', '_')}_{args.year}_data.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} in dir: {dataset_dir}")

    return df

if __name__ == "__main__":
    main()