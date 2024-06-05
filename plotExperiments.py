# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Importing user-defined modules


def plottotal(data_path, df_tot):

    # Create a figure and subplots
    fig, axis = plt.subplots(2, 2, sharex = False, figsize = (20, 10))
    fig.suptitle('Diameter comparison for different roughness values')

    # Get unique roughness values
    unique_roughness = df_tot['roughness'].unique()

    # Loop over the unique values of roughness and plot the diameter in both directions scaled by the initial diameter
    for i, roughness in enumerate(unique_roughness):
        # Filter the DataFrame to only include rows with that roughness value
        df_filtered = df_tot[df_tot['roughness'] == roughness]

        """ color = 'bo' if roughness == 0 else 'ro' if roughness == 4 else 'go'
        print('color: ', color, type(color))
        print('roughness: ', roughness, type(roughness))
        print("bools: ", roughness == 0, roughness == 2, roughness == 4) """

        # Plot data roughness direction each
        axis[i//2, i%2].plot(df_filtered['height']/(df_filtered['diameter']/2), 
                            (df_filtered['d_vertical']-df_filtered['diameter'])/df_filtered['diameter'], 
                            'bo' if roughness == 0 else 'ro' if roughness == 4 else 'go', 
                            label='parallel to roughness')
        axis[i//2, i%2].plot(df_filtered['height']/(df_filtered['diameter']/2), 
                            (df_filtered['d_horizontal']-df_filtered['diameter'])/df_filtered['diameter'], 
                            'bs' if roughness == 0 else 'rs' if roughness == 4 else 'gs', 
                            label='perpendicular to roughness')
        
        # Plot the total diameter comparison
        axis[1, 1].plot(df_filtered['height']/(df_filtered['diameter']/2),
                        np.where(df_filtered['direction'] == 'pa', df_filtered['d_horizontal']/df_filtered['d_vertical'], df_filtered['d_vertical']/df_filtered['d_horizontal']),
                        'b^' if roughness == 0 else 'r^' if roughness == 4 else 'g^',	 
                        label=f'{roughness} mm roughness perpendicular')
        
        # Get the minimum and maximum x-values
        x_min = min(df_filtered['height']/(df_filtered['diameter']/2))
        x_max = max((df_filtered['height']/(df_filtered['diameter']/2)))

        # Create an array of x-values for the guide functions
        x_values_1 = np.linspace(x_min, 1.7, 100)
        x_values_2 = np.linspace(1.7, x_max, 100)
        
        # Define the guide functions
        # Define your guide functions
        guide_func1 = lambda x: 1.24 * x
        guide_func2 = lambda x: 1.6 * np.sqrt(x)

        # Calculate the y-values for the guide functions
        y_values1 = guide_func1(x_values_1)
        y_values2 = guide_func2(x_values_2)

        # Plot the guide functions
        axis[i//2, i%2].plot(x_values_1, y_values1, color='gray', linestyle=':', label='Guide functions')
        axis[i//2, i%2].plot(x_values_2, y_values2, color='gray', linestyle=':')

        # Set x and y axes to logarithmic scale
        axis[i//2, i%2].set_xscale('log')
        axis[i//2, i%2].set_yscale('log')

        """ # Do the same for the axis[1, 1] subplot
        axis[1, 1].set_xscale('log')
        axis[1, 1].set_yscale('log') """

        # Set plot title and labels
        axis[i//2, i%2].set_title(f'Roughness: {roughness} mm')
        axis[i//2, i%2].set_xlabel('aspect ratio = initial height / initial radius')
        axis[i//2, i%2].set_ylabel('(final radius - initial \nradius) / initial radius')
        axis[i//2, i%2].legend()

    # Set plot title and labels
    axis[1, 1].set_title(f'Roughness comparsion')
    axis[1, 1].set_xlabel('aspect ratio = initial height / initial radius')
    axis[1, 1].set_ylabel('final radius\northogonal to roughness / final radius\nparallel to roughness')
    axis[1, 1].legend()

    # Make directory for graphs and save plot
    try:
        os.mkdir(f'{data_path}graphs/')
        print('Directory graphs created and plot saved as pdf.')
    except FileExistsError:
        print('Directory graphs already exists but plot saved as pdf.')

    # Save the figure before showing it
    plt.savefig(f'{data_path}graphs/total.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0.1, orientation = 'landscape')

    # Then display the figure
#    plt.show()
    
    return

if __name__ == "__main__":

    # Define path with data to be analysed with artificial data
#    data_path = 'G:/data/tests/sony_hs/camera/'
#    df_tot = pd.read_csv('G:/data/tests/sony_hs/camera/raw_data/total_raw - Kopie.csv')

    # Define path with data to be analysed with real data - from one test day with defined folder structure
#    data_path = 'G:/experiments/combined/'
#    df_tot = pd.read_csv(f'{data_path}camera/raw_data/total_raw.csv')

    # Define the data path for analysing multiple test days from different csv files
    data_path = 'G:/experiments/total/'

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(f'{data_path}*.csv')

    # Initialize an empty list to store the dataframes
    dfs = []

    # Loop through the list of CSV files
    for i, csv_file in enumerate(csv_files):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file, usecols=lambda x: x != 0)
        # Append the DataFrame to the list
        dfs.append(df)

    print(f'dfs: {dfs}')

    # Concatenate all the dataframes in the list into a single DataFrame
    df_tot = pd.concat(dfs, ignore_index=True)

    # Sort df_tot by 'roughness' and 'diameter' in increasing order
    df_tot = df_tot.sort_values(by=['roughness', 'diameter'])

    print('df_tot: ', df_tot)

    plottotal(data_path, df_tot)
