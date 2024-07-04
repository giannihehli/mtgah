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

    # Get unique diameter values
    unique_diameter = df_tot['diameter'].unique()

    # Get the minimum and maximum x-values
    x_min = 0.7 * min(df_tot['height']/(df_tot['diameter']/2))
    x_max = 1.5 * max((df_tot['height']/(df_tot['diameter']/2)))

    # Set the x-axis limits
    axis[0, 0].set_xlim([x_min, x_max])
    axis[0, 1].set_xlim([x_min, x_max])
    axis[1, 0].set_xlim([x_min, x_max])
    axis[1, 1].set_xlim([x_min, x_max])

    # Loop over the unique values of roughness and plot the diameter in both directions 
    #  scaled by the initial diameter
    for i, roughness in enumerate(unique_roughness):
        # Filter the DataFrame to only include rows with that roughness value
        df_filtered = df_tot[df_tot['roughness'] == roughness]

        # Plot data roughness direction each
        axis[i//2, i%2].plot(df_filtered['height']/(df_filtered['diameter']/2), 
                            (df_filtered['d_vertical']-df_filtered['diameter'])
                            /df_filtered['diameter'], 
                            'bo' if roughness == 0 else 'go' if roughness == 2 else 'ro', 
                            label='parallel to roughness')
        axis[i//2, i%2].plot(df_filtered['height']/(df_filtered['diameter']/2), 
                            (df_filtered['d_horizontal']-df_filtered['diameter'])
                            /df_filtered['diameter'], 
                            'b^' if roughness == 0 else 'g^' if roughness == 2 else 'r^', 
                            label='perpendicular to roughness')   

        # Create an array of x-values for the guide functions
        x_values_1 = np.linspace(x_min, 1.7, 100)
        x_values_2 = np.linspace(1.7, x_max, 100)
        
        # Define the guide functions
        guide_func1 = lambda x: 1.24 * x
        guide_func2 = lambda x: 1.6 * np.sqrt(x)

        # Calculate the y-values for the guide functions
        y_values1 = guide_func1(x_values_1)
        y_values2 = guide_func2(x_values_2)

        # Plot the guide functions
        axis[i//2, i%2].plot(x_values_1, y_values1, color='gray', linestyle=':', 
                             label='Guide function g(a)')
        axis[i//2, i%2].plot(x_values_2, y_values2, color='gray', linestyle=':')
        axis[1, 1].axhline(y=1, color='gray', linestyle=':')

        # Set x and y axes to logarithmic scale
        axis[i//2, i%2].set_xscale('log')
        axis[i//2, i%2].set_yscale('log')

        # Do the same for the axis[1, 1] subplot
        axis[1, 1].set_xscale('log')
        
        # Set plot title and labels
        axis[i//2, i%2].set_title(f'Roughness: {roughness} mm')
        axis[i//2, i%2].set_xlabel('aspect ratio = initial height / initial radius')
        axis[i//2, i%2].set_ylabel('(final radius - initial \nradius) / initial radius')
        axis[i//2, i%2].legend(loc='lower right', fontsize = 10)

        for diameter in unique_diameter:
            # Filter the DataFrame to only include rows with that diameter value
            df_filtered_diameter = df_filtered[df_filtered['diameter'] == diameter]
        
            # Define the plotting style
            marker_color = ('bD' if roughness == 0 and diameter == unique_diameter[0]
                    else 'gD' if roughness == 2 and diameter == unique_diameter[0] 
                    else 'rD' if roughness == 4 and diameter == unique_diameter[0]
                    else 'bp' if roughness == 0 and diameter == unique_diameter[1]
                    else 'gp' if roughness == 2 and diameter == unique_diameter[1]
                    else 'rp' if roughness == 4 and diameter == unique_diameter[1]
                    else 'bH' if roughness == 0
                    else 'gH' if roughness == 2
                    else 'rH')
            
            # Plot the total diameter comparison
            axis[1, 1].plot(df_filtered_diameter['height']/
                            (df_filtered_diameter['diameter']/2),
                            np.where(df_filtered_diameter['direction'] == 'pa', 
                                     df_filtered_diameter['d_horizontal']
                                     /df_filtered_diameter['d_vertical'], 
                                     df_filtered_diameter['d_vertical']
                                     /df_filtered_diameter['d_horizontal']),
                            marker_color)

    # Define the plotting style and labels of the lower right plot
    colors = ['b', 'g', 'r']
    shapes = ['D', 'p', 'H']
    labels1 = [f'{roughness} mm roughness' for roughness in unique_roughness]
    labels2 = [f'{diameter} mm diameter' for diameter in unique_diameter]
     
    # Create handles
    handles1 = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                           markersize=8) for color in colors]
    handles2 = [plt.Line2D([0], [0], marker=shape, color='w', markerfacecolor='gray', 
                           markersize=8) for shape in shapes]

    # Create the legends
    legend1 = axis[1, 1].legend(handles=handles1, labels=labels1, loc='lower right', fontsize = 10)
    legend2 = axis[1, 1].legend(handles=handles2, labels=labels2, loc='lower left', fontsize = 10)

    # Add the legends manually to the current Axes.
    axis[1, 1].add_artist(legend1)
    axis[1, 1].add_artist(legend2)

    # Set plot title and labels
    axis[1, 1].set_title(f'Roughness comparison')
    axis[1, 1].set_xlabel('aspect ratio = initial height / initial radius')
    axis[1, 1].set_ylabel('final radius\nperpendicular / final radius\nparallel to roughness')
    axis[1, 1].legend(fontsize = 10)

    # Make directory for graphs and save plot
    try:
        os.mkdir(f'{data_path}graphs/')
        print('Directory graphs created and plot saved as pdf.')
    except FileExistsError:
        print('Directory graphs already exists but plot saved as pdf.')

    # Save the figure before showing it
    plt.savefig(f'{data_path}graphs/total.pdf', transparent = True, 
                bbox_inches = 'tight', pad_inches = 0.1, orientation = 'landscape')

    # Then display the figure
#    plt.show()
    
    return

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define the data path for analysing multiple test days from different csv files
    data_path = 'G:/horizontal experiments/combined/'

    ####################################################################################

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(f'{data_path}data/*.csv')

    # Initialize an empty list to store the dataframes
    dfs = []

    # Loop through the list of CSV files
    for i, csv_file in enumerate(csv_files):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file, usecols=lambda x: x != 0)
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all the dataframes in the list into a single DataFrame
    df_tot = pd.concat(dfs, ignore_index=True)

    # Sort df_tot by 'roughness' and 'diameter' in increasing order
    df_tot = df_tot.sort_values(by=['roughness', 'diameter'])

    plottotal(data_path, df_tot)