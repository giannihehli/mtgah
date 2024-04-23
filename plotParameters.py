# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing user-defined modules


def plot(df, vid):
    # Define subplots
    fig, axis = plt.subplots(2, 2, sharex = False, figsize = (1000, 1000))
    fig.suptitle(vid, fontsize=16)

    # Plot radius parameters at top right
    axis[0, 0].set_title('Distance')
    axis[0, 0].plot(df['time'], df['distance_right'], label = 'Distance Right [mm]', color = 'blue')
    axis[0, 0].plot(df['time'], df['distance_left'], label = 'Distance Left [mm]', color = 'green')
    axis[0, 0].plot(df['time'], df['distance_bottom'], label = 'Distance Bottom [mm]', color = 'orange')
    axis[0, 0].set_xlabel('Time [s]')
    axis[0, 0].set_ylabel('Radius [mm]')
    axis[0, 0].legend()

    # Plot velocity parameters at top left
    axis[0, 1].set_title('Velocity')
    axis[0, 1].plot(df['time'], df['velocity_right'], label = 'Velocity Right [mm/s]', color = 'blue')
    axis[0, 1].plot(df['time'], df['velocity_left'], label = 'Velocity Left [mm/s]', color = 'green')
    axis[0, 1].plot(df['time'], df['velocity_bottom'], label = 'Velocity Bottom [mm/s]', color = 'orange')
    axis[0, 1].set_xlabel('Time [s]')
    axis[0, 1].set_ylabel('Velocity [mm/s]')
    axis[0, 1].legend()

    # Plot radius ratio at bottom right
    axis[1, 0].set_title('Ratio Horizontal vs. Vertical Radius')
    axis[1, 0].plot(df['distance_right'], df['distance_bottom'], label = 'Distance Right [mm]', color = 'blue')
    axis[1, 0].plot(df['distance_left'], df['distance_bottom'], label = 'Distance Left [mm]', color = 'green')
    axis[1, 0].set_xlabel('Horizontal Distance [mm]')
    axis[1, 0].set_ylabel('Vertical Distance [mm]')
    axis[1, 0].set_aspect('equal', 'box')
    axis[1, 0].legend()

    # Plot velocity ratio at bottom left
    axis[1, 1].set_title('Ratio Horizontal vs. Vertical Velocity')
    axis[1, 1].plot(df['velocity_right'], df['velocity_bottom'], label = 'Velocity Right[mm/s]', color = 'blue')
    axis[1, 1].plot(df['velocity_left'], df['velocity_bottom'], label = 'Velocity Left [mm/s]', color = 'green')
    axis[1, 1].set_xlabel('Horizontal Velocity [mm/s]')
    axis[1, 1].set_ylabel('Vertical Velocity [mm/s]')
    axis[1, 1].set_aspect('equal', 'box')
    axis[1, 1].legend()

    plt.show()

    return

if __name__ == "__main__":
    # Import measured parameters
    df = pd.read_csv('H:/data/tests/sony_hs/f_r8_d113_h40.csv')

    plot(df, 'f_r8_d113_h40')