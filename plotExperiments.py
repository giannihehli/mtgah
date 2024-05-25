# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

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
    
        # Filter data for 'pa' direction
        df_pa = df_filtered[df_filtered['direction'] == 'pa']
        axis[i//2, i%2].plot(df_pa['height']/(df_pa['diameter']/2), 
                            0.1*df_pa['d_vertical']/df_pa['diameter'], 
                            'bo' if roughness == 0 else 'ro' if roughness == 4 else 'go', 
                            label='parallel to roughness',
                            fillstyle = 'none')

        # Filter data for 'pe' direction
        df_pe = df_filtered[df_filtered['direction'] == 'pe']
        axis[i//2, i%2].plot(df_pe['height']/(df_pe['diameter']/2), 
                            0.1*df_pe['d_horizontal']/df_pe['diameter'], 
                            'bx' if roughness == 0 else 'rx' if roughness == 4 else 'gx', 
                            label='perpendicular to roughness',
                            fillstyle = 'none')
        
        axis[1, 1].plot(df_filtered['height']/(df_filtered['diameter']/2),
                        np.where(df_filtered['direction'] == 'pa', df_filtered['d_horizontal']/df_filtered['d_vertical'], df_filtered['d_vertical']/df_filtered['d_horizontal']),
                        'b^' if roughness == 0 else 'r^' if roughness == 4 else 'g^',	 
                        label=f'{roughness} mm roughness perpendicular')
    
        axis[i//2, i%2].set_title(f'Roughness: {roughness} mm')
        axis[i//2, i%2].set_xlabel('aspect ratio = initial height / initial radius')
        axis[i//2, i%2].set_ylabel('measured diameter / initial diameter')
        axis[i//2, i%2].legend()
    
    axis[1, 1].set_title(f'Roughness comparsion')
    axis[1, 1].set_xlabel('aspect ratio = initial height / initial radius')
    axis[1, 1].set_ylabel('final diameter\northogonal to roughness / final diameter\nparallel to roughness')
    axis[1, 1].legend()

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
