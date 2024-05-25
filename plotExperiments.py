# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Importing user-defined modules


def plottotal(data_path, df_tot):

    # Create a figure and subplots
    fig, axis = plt.subplots(2, 2)
    fig.suptitle('Diameter comparison for different roughness values')

    # Get unique roughness values
    unique_roughness = df_tot['roughness'].unique()

    # Loop over the unique values of roughness and plot the diameter in both directions scaled by the initial diameter
    for i, roughness in enumerate(unique_roughness):
        # Filter the DataFrame to only include rows with that roughness value
        df_filtered = df_tot[df_tot['roughness'] == roughness]
    
        # Filter data for 'pa' direction
        df_pa = df_filtered[df_filtered['direction'] == 'pa']
        axis[i//2, i%2].plot(df_pa['height']/(2*df_pa['diameter']), 
                            df_pa['d_vertical']/df_pa['diameter'], 
                            'o', label='parallel diameter / initial diameter')
    
        # Filter data for 'non-pa' direction
        df_non_pa = df_filtered[df_filtered['direction'] != 'pa']
        axis[i//2, i%2].plot(df_non_pa['height']/(2*df_non_pa['diameter']), 
                            df_non_pa['d_horizontal']/df_non_pa['diameter'], 
                            's', label='perpendicular diameter / initial diameter')
    
        axis[i//2, i%2].set_title(f'Roughness: {roughness}')
        axis[i//2, i%2].legend()

    # Filter data for different roughness values
    df_r0 = df_tot[df_tot['roughness'] == '0']
    df_r4 = df_tot[df_tot['roughness'] == '4']
    df_r8 = df_tot[df_tot['roughness'] == '8']

    # Plot the ratio of the two diameters for no roughness
    axis[1, 1].plot(df_r0['height']/(2*df_r0['diameter']),
                   np.where(df_r0['direction'] == 'pa', df_r0['d_horizontal']/df_r0['d_vertical'], df_r0['d_vertical']/df_r0['d_horizontal']),
                   'o', label='No roughness')
    
    # Plot the ratio of the two diameters for roughness 4 mm
    axis[1, 1].plot(df_r4['height']/(2*df_r4['diameter']),
                   np.where(df_r4['direction'] == 'pa', df_r4['d_horizontal']/df_r4['d_vertical'], df_r4['d_vertical']/df_r4['d_horizontal']),
                   '^', label='4 mm roughness')
    
    # Plot the ratio of the two diameters for roughness 8 mm
    axis[1, 1].plot(df_r8['height']/(2*df_r8['diameter']),
            np.where(df_r8['direction'] == 'pa', df_r8['d_horizontal']/df_r8['d_vertical'], df_r8['d_vertical']/df_r8['d_horizontal']),
            'p', label='8 mm roughness')
    
    axis[1, 1].set_title(f'Roughness comparsion')
    axis[1, 1].legend()

    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig(f'{data_path}graphs/total.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0.1, orientation = 'landscape')

    # Then display the figure
    plt.show()
    
    return

if __name__ == "__main__":

    # Define path with data to be analysed
    data_path = 'H:/data/tests/sony_hs/'

    # Import measured parameters
    df_tot = pd.read_csv('H:/data/tests/sony_hs/raw_data/total_raw - Kopie.csv')

    plottotal(data_path, df_tot)
