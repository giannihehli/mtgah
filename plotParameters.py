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
    axis[0, 0].set_title('Radius')
    axis[0, 0].plot(df['time'], df['horizontal_radius'], label = 'Horizontal Radius [mm]', color = 'blue')
    axis[0, 0].plot(df['time'], df['vertical_radius'], label = 'Vertical Radius [mm]', color = 'orange')
    axis[0, 0].set_xlabel('Time [s]')
    axis[0, 0].set_ylabel('Radius [mm]')
    axis[0, 0].legend()

    # Plot velocity parameters at top left
    axis[0, 1].set_title('Velocity')
    axis[0, 1].plot(df['time'], df['horizontal_velocity'], label = 'Horizontal Velocity [mm/s]', color = 'blue')
    axis[0, 1].plot(df['time'], df['vertical_velocity'], label = 'Vertical Velocity [mm/s]', color = 'orange')
    axis[0, 1].set_xlabel('Time [s]')
    axis[0, 1].set_ylabel('Velocity [mm/s]')
    axis[0, 1].legend()

    # Plot radius comparsion at bottom right
    axis[1, 0].set_title('Horizontal vs. Vertical Radius')
    axis[1, 0].plot(df['horizontal_radius'], df['vertical_radius'], label = 'Radius [mm]', color = 'blue')
    axis[1, 0].set_xlabel('Horizontal Radius [mm]')
    axis[1, 0].set_ylabel('Vertical Radius [mm]')
    axis[1, 0].set_aspect('equal', 'box')

    # Plot velocity comparsion at bottom left
    axis[1, 1].set_title('Horizontal vs. Vertical Velocity')
    axis[1, 1].plot(df['horizontal_velocity'], df['vertical_velocity'], label = 'Velocity [mm/s]', color = 'blue')
    axis[1, 1].set_xlabel('Horizontal Velocity [mm/s]')
    axis[1, 1].set_ylabel('Vertical Velocity [mm/s]')
    axis[1, 1].set_aspect('equal', 'box')

    plt.show()
    
    """ plt.plot(df['time'], df['horizontal_radius'], label = 'Horizontal Radius [mm]', color = 'blue')
    plt.plot(df['time'], df['vertical_radius'], label = 'Vertical Radius [mm]', color = 'orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Radius [mm]')
    plt.legend()
    plt.show() """

#    plt.plot(df['time'], df['horizontal_velocity'], label = 'Horizontal Velocity [mm/s]', color = 'blue')


if __name__ == "__main__":
    # Import measured parameters
    df = pd.read_csv('H:/data/tests/sony_hs/f_r8_d113_h40.csv')

    plot(df, 'f_r8_d113_h40')