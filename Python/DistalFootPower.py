# -*- coding: utf-8 -*-
"""
Distal segment power computation for treadmill locomotion.

Commonly applied to the rearfoot to obtain distal rearfoot power. The power
formulation was provided in Takahashi et al. 2012; originally in Siegel et al.
1996. For full derivations, see Zelik and Honert 2018 appendix.

Extracted from Performance_Walk_ExtractKinematicKineticVars.py for modularity.
"""

import numpy as np
import matplotlib.pyplot as plt


def dist_seg_power_treadmill(Seg_COM_Pos, Seg_COM_Vel, Seg_Ang_Vel,
                              CenterOfPressure, GRF, FreeMoment, speed,
                              landings, takeoffs, yn_run):
    """
    The purpose of this function is to compute the distal segment power -
    commonly applied to the rearfoot to obtain the distal rearfoot power. The
    power in this formation was provided in Takahashi et al. 2012; but
    originally in Siegel et al. 1996. For full derivations, see Zelik and
    Honert 2018 appendix. This code assumes that the locomotion direction is +y

    Parameters
    ----------
    Seg_COM_Pos : numpy array (N X 3)
        Segment COM Position (ex: Foot COM Position)
    Seg_COM_Vel : numpy array (N X 3)
        Segment COM Velocity (ex: Foot COM Velocity)
    Seg_Ang_Vel : numpy array (N X 3)
        Segment Angular Velocity (ex: Foot Angular Velocity)
    CenterOfPressure : numpy array (N X 3)
        Location of the center of pressure
    GRF : numpy array (N X 3)
        Ground Reaction Force. Ensure that the input GRF is the REACTION
    FreeMoment : numpy array (N X 3)
        Free moment on force platform
    speed : float
        Treadmill belt speed - can be used as a debugging variable or to set
        the speed of the foot in 3D space.
    landings : list
        Initial foot contact: used only during walking
    takeoffs : list
        Or toe-offs: used only during walking
    yn_run : int
        1 for running, 0 for walking

    Returns
    -------
    power : numpy array
        distal rearfoot power

    """

    # If walking: compute the treadmill belt speed
    if yn_run == 0:
        # Debugging variable to examine foot speed
        debug = 0
        # When using a treadmill is used for locomotion and the distal segment
        # power is computed, the treadmill belt speed needs to be taken into
        # account. Based on prior experience, DURING WALKING, foot flat can provide
        # a decent approximation of the treadmill belt speed.
        foot_flat = [0.2,0.4]

        # Allocate variables
        step_speed = np.zeros((len(landings)-1,1))

        # Index through the landings
        for ii in range(len(landings)-1):
            stepframe = takeoffs[ii]-landings[ii]
            # Frames to analyze based on the foot flat percentages
            FFframes = range(landings[ii]+round(foot_flat[0]*stepframe),landings[ii]+round(foot_flat[1]*stepframe),1)
            step_speed[ii] = np.mean(Seg_COM_Vel[FFframes,1])

        # Find the average treadmill belt speed of the trial (also exclude any
        # zeros in the estimate)
        avg_speed = -np.mean(step_speed[step_speed != 0])

        if debug == 1:
            plt.figure(1010)
            plt.plot(step_speed)

    # If running: use the inputted Bertec treadmill speed
    else:
        # It is difficult to compute the belt speed from running - thus rely on
        # the set treadmill belt speed
        avg_speed = np.array(speed)

    # Treadmill belt speed: will need to be updated based on the slope
    belt_vel = np.array(list(zip([0]*len(Seg_COM_Vel),[avg_speed]*len(Seg_COM_Vel),[0]*len(Seg_COM_Vel))))

    # Adjust the segment velocity based on belt speed
    adj_Seg_COM_Vel = Seg_COM_Vel+belt_vel
    # Compute the rotational and translational components of the power
    power_rot = np.sum(np.cross(CenterOfPressure-Seg_COM_Pos,GRF,axis=1)*Seg_Ang_Vel,axis=1)+np.sum(FreeMoment*Seg_Ang_Vel,axis=1)
    power_tran = np.sum(GRF*adj_Seg_COM_Vel,axis=1)

    power = power_rot+power_tran
    return power
