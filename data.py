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
    Get each driver fastest lap for a conventional event (FP1, FP2, FP3, Q, R)

    returns:
    dataframe of all drivers fastest laps for the event
    '''
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    final_results = {}

    for session_type in session_types:
        try:
            session = ff1.get_session(year, event_name, session_type)
            session.load()
        
            drivers = session.laps['Driver'].unique()
            for driver in drivers:
                if driver not in final_results:
                    final_results[driver] = {
                        'driver_number': session.laps.loc[session.laps['Driver'] == driver, 'DriverNumber'].iloc[0],
                        'year': year,
                        'event': event_name,
                        'team': session.laps.loc[session.laps['Driver'] == driver, 'Team'].iloc[0]
                    }
                driver_laps = session.laps.pick_driver(driver)
                    
                if not driver_laps.empty:
                    sector_times = driver_laps[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean().dropna()             
                    total_times = sector_times.sum().total_seconds() if not sector_times.isna().any() else None

                    final_results[driver] = {
                        **final_results[driver],
                        f'{session_type}_sector1_time': sector_times['Sector1Time'].total_seconds() if 'Sector1Time' in sector_times else None,
                        f'{session_type}_sector2_time': sector_times['Sector2Time'].total_seconds() if 'Sector2Time' in sector_times else None,
                        f'{session_type}_sector3_time': sector_times['Sector3Time'].total_seconds() if 'Sector3Time' in sector_times else None,
                        f'{session_type}_total_time': total_times
                    }
            
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