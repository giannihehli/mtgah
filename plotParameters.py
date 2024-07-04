# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing user-defined modules


def plotparams(data_path, exp, df, layout, basis, direction, d, height, 
               d_vertical, d_horizontal):

    # Define roughness basis
    basis = basis[1]
    
    # Define plot labels according to direction
    match direction:
        case 'pa':
            distance_right = 'Distance perpendicular right [mm]'
            distance_left = 'Distance perpendicular left [mm]'
            distance_bottom = 'Distance parallel bottom [mm]'
            distance_horizontal = 'Distance perpendicular [mm]'
            velocity_right = 'Velocity perpendicular right [mm/s]'
            velocity_left = 'Velocity perpendicular left [mm/s]'
            velocity_bottom = 'Velocity parallel bottom [mm/s]'
            velocity_horizontal = 'Velocity perpendicular [mm/s]'
            horizontal_distance = 'Perpendicular distance [mm]'
            vertical_distance = 'Parallel distance [mm]'
            horizontal_velocity = 'Perpendicular velocity [mm/s]'
            vertical_velocity = 'Parallel velocity [mm/s]'
            horizontal_diameter = 'perpendicular diameter'
            vertical_diameter = 'parallel diameter'
            horizontal_radius = 'Perpendicular radius [mm]'
            vertical_radius = 'Parallel radius [mm]'
        case 'pe':
            distance_right = 'Distance parallel right [mm]'
            distance_left = 'Distance parallel left [mm]'
            distance_bottom = 'Distance perpendicular bottom [mm]'
            distance_horizontal = 'Distance parallel [mm]'
            velocity_right = 'Velocity parallel right [mm/s]'
            velocity_left = 'Velocity parallel left [mm/s]'
            velocity_bottom = 'Velocity perpendicular bottom [mm/s]'
            velocity_horizontal = 'Velocity parallel [mm/s]'
            horizontal_distance = 'Parallel distance [mm]'
            vertical_distance = 'Perpendicular distance [mm]'
            horizontal_velocity = 'Parallel velocity [mm/s]'
            vertical_velocity = 'Perpendicular velocity [mm/s]'
            horizontal_diameter = 'parallel diameter'
            vertical_diameter = 'perpendicular diameter'
            horizontal_radius = 'Parallel radius [mm]'
            vertical_radius = 'Perpendicular radius [mm]'

    # Define subplots
    fig, axis = plt.subplots(2, 3, sharex = False, figsize = (30, 10))
    fig.suptitle(f'\n {layout} mount with {basis} mm roughness basis, {d} mm cylinder and {height} mm sand height', fontsize=16)
    fig.text(0.5, -0.01, f'Final {horizontal_diameter}: {round(d_horizontal, 1)} mm \n Final {vertical_diameter}: {round(d_vertical, 1)} mm \n', ha='center')
    plt.rc('legend',fontsize=5) # using a size in points

    # Plot distance parameters at top left
    axis[0, 0].set_title('Distance')
    axis[0, 0].plot(df['time'], df['distance_right'], 
                    label = distance_right, color = 'blue')
    axis[0, 0].plot(df['time'], df['distance_left'], 
                    label = distance_left, color = 'orange')
    axis[0, 0].plot(df['time'], df['distance_horizontal'], 
                    label = distance_horizontal, color = 'green')
    axis[0, 0].plot(df['time'], df['distance_bottom'], 
                    label = distance_bottom, color = 'red')
    axis[0, 0].set_xlabel('Time [s]')
    axis[0, 0].set_ylabel('Distance from first measurement [mm]')
    axis[0, 0].legend(loc='lower right', fontsize = 10)

    # Plot velocity parameters at top middle
    axis[0, 1].set_title('Velocity')
    axis[0, 1].plot(df['time'], df['velocity_right'], 
                    label = velocity_right, color = 'blue')
    axis[0, 1].plot(df['time'], df['velocity_left'], 
                    label = velocity_left, color = 'orange')
    axis[0, 1].plot(df['time'], df['velocity_horizontal'], 
                    label = velocity_horizontal, color = 'green')
    axis[0, 1].plot(df['time'], df['velocity_bottom'], 
                    label = velocity_bottom, color = 'red')
    axis[0, 1].set_xlabel('Time [s]')
    axis[0, 1].set_ylabel('Velocity [mm/s]')
    axis[0, 1].legend(loc='upper right', fontsize = 10)

    # Plot radius parameters at top right
    axis[0, 2].set_title('Radius')
    axis[0, 2].plot(df['time'], df['radius_horizontal'], 
                    label = horizontal_radius, color = 'green')
    axis[0, 2].plot(df['time'], df['radius_vertical'], 
                    label = vertical_radius, color = 'red')
    axis[0, 2].set_xlabel('Time [s]')
    axis[0, 2].set_ylabel('Radius [mm]')
    axis[0, 2].legend(loc='lower right', fontsize = 10)

    # Plot radius ratio at bottom left
    axis[1, 0].set_title('Ratio Horizontal vs. Vertical Distance')
    axis[1, 0].plot(df['distance_bottom'], df['distance_right'], 
                    label = distance_right, color = 'blue')
    axis[1, 0].plot(df['distance_bottom'], df['distance_left'], 
                    label = distance_left, color = 'orange')
    axis[1, 0].plot(df['distance_bottom'], df['distance_horizontal'], 
                    label = distance_horizontal, color = 'green')
    axis[1, 0].plot(df['distance_bottom'], df['distance_bottom'], '--', 
                    label = 'ratio 1:1', color = 'grey')
    axis[1, 0].set_ylabel(horizontal_distance)
    axis[1, 0].set_xlabel(vertical_distance)
    axis[1, 0].set_aspect('equal', 'box')
    axis[1, 0].legend(loc='lower right', fontsize = 7)

    # Plot velocity ratio at bottom middle
    axis[1, 1].set_title('Ratio Horizontal vs. Vertical Velocity')
    axis[1, 1].plot(df['velocity_bottom'], df['velocity_right'], 
                    label = velocity_right, color = 'blue')
    axis[1, 1].plot(df['velocity_bottom'], df['velocity_left'], 
                    label = velocity_left, color = 'orange')
    axis[1, 1].plot(df['velocity_bottom'], df['velocity_horizontal'], 
                    label = velocity_horizontal, color = 'green')
    axis[1, 1].plot(df['velocity_bottom'], df['velocity_bottom'], '--', 
                    label = 'ratio 1:1', color = 'grey')
    axis[1, 1].set_ylabel(horizontal_velocity)
    axis[1, 1].set_xlabel(vertical_velocity)
    axis[1, 1].set_aspect('equal', 'box')
    axis[1, 1].legend(loc='lower right', fontsize = 7)

    # Plot radius ratio at bottom right
    axis[1, 2].set_title('Ratio Horizontal vs. Vertical Radius')
    axis[1, 2].plot(df['radius_vertical'], df['radius_horizontal'], 
                    label = horizontal_radius, color = 'green')
    axis[1, 2].plot(df['radius_vertical'], df['radius_vertical'], '--', 
                    label = vertical_radius, color = 'red')
    axis[1, 2].set_ylabel(horizontal_radius)
    axis[1, 2].set_xlabel(vertical_radius)
    axis[1, 2].set_aspect('equal', 'box')
    axis[1, 2].legend(loc='lower right', fontsize = 7)

    # Make directory for graphs and save plot
    try:
        os.mkdir(f'{data_path}graphs/')
    except FileExistsError:
        pass
    
    plt.savefig(f'{data_path}graphs/{exp}.pdf', transparent = True, bbox_inches = 'tight', 
                pad_inches = 0.1, orientation = 'landscape')
    
    # Show plot
#    plt.show()
    
    return

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define paths
    data_path = 'data/'
    data = 'f_r2-pa_d41_h148_16_raw.csv'

    ####################################################################################

    # Import measured parameters
    df = pd.read_csv(data_path + data)

    # Define video name
    exp = data.split('.')[0]

    # Define used parameters according to video name
    layout = exp.split('_')[0]
    basis = exp.split('_')[1]
    roughness = basis[1]
    direction = basis.split('-')[1]
    diameter = exp.split('_')[2]
    height = exp.split('_')[3]
    exp_id = exp.split('_')[4]

    # Define diameter of final deposit
    diameter_vertical = df['diameter_vertical'].iloc[-1]
    diameter_horizontal = df['diameter_horizontal'].iloc[-1]

    # Define initial height according to height
    h_initial = int(height[1:])

    # Plot parameters
    plotparams(data_path, exp, df, layout, basis, direction, diameter[1:], h_initial, 
               diameter_vertical, diameter_horizontal)