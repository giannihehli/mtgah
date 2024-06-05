# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                            (0.1*df_filtered['d_vertical']-df_filtered['diameter'])/df_filtered['diameter'], 
                            'bo' if roughness == 0 else 'ro' if roughness == 4 else 'go', 
                            label='parallel to roughness')
        axis[i//2, i%2].plot(df_filtered['height']/(df_filtered['diameter']/2), 
                            (0.1*df_filtered['d_horizontal']-df_filtered['diameter'])/df_filtered['diameter'], 
                            'bs' if roughness == 0 else 'rs' if roughness == 4 else 'gs', 
                            label='perpendicular to roughness')
        
        # Plot the total diameter comparison
        axis[1, 1].plot(df_filtered['height']/(df_filtered['diameter']/2),
                        np.where(df_filtered['direction'] == 'pa', df_filtered['d_horizontal']/df_filtered['d_vertical'], df_filtered['d_vertical']/df_filtered['d_horizontal']),
                        'b^' if roughness == 0 else 'r^' if roughness == 4 else 'g^',	 
                        label=f'{roughness} mm roughness perpendicular')
        
        # Get the minimum and maximum x-values
        x_min = df_filtered['height'].min() / (df_filtered['diameter'].min() / 2)
        x_max = df_filtered['height'].max() / (df_filtered['diameter'].max() / 2)

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

    # Save the figure before showing it
    plt.savefig(f'{data_path}graphs/total.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0.1, orientation = 'landscape')

    # Then display the figure
    plt.show()
    
    return

if __name__ == "__main__":

    # Define path with data to be analysed with artificial data
#    data_path = 'G:/data/tests/sony_hs/camera/'
#    df_tot = pd.read_csv('G:/data/tests/sony_hs/camera/raw_data/total_raw - Kopie.csv')

    # Define path with data to be analysed with real data
    data_path = 'G:/experiments/20240531/camera/'
    df_tot = pd.read_csv(f'{data_path}raw_data/total_raw.csv')

    plottotal(data_path, df_tot)
