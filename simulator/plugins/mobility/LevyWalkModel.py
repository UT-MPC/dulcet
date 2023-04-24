# credit to Sang Su Lee: https://github.com/UT-MPC/swarm/blob/main/simulate_levy.py
import pandas as pd
import numpy as np

from scipy.stats import uniform
from scipy.stats import levy

def truncated_levy(n, thres, alpha):
    r = np.zeros(1)

    while r.size < n:
        r = levy.rvs(size=(int)(2 * n))
        r = np.power(r, -(1+alpha))
        r = r[r < thres]
        r = r[:n]

    return r

def time_to_tread_path(p1, p2, k, rho):
    x = np.linalg.norm(p1 - p2)
    return k * np.power(x, (1-rho))

def levy_walk_not_interpolated(n, bounds, x_orig, y_orig, thres_loc, thres_time, alpha, beta):
    K = 1.
    RHO = .5

    # uniformly distributed angles
    angle = uniform.rvs(size=(n,), loc=.0, scale=2.*np.pi)

    # levy distributed step length
    rv = truncated_levy(n, thres_loc, alpha)
    stay_time = truncated_levy(n, thres_time, beta)

    # x and y coordinates
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x_orig
    y[0] = y_orig

    for i in range(1,n):
        x[i] = x[i-1] + rv[i] * np.cos(angle[i])
        if x[i] > bounds[0]:
            x[i] -= 2 * (x[i] - bounds[0])
        elif x[i] < -bounds[0]:
            x[i] -= 2 * (x[i] + bounds[0])
            
        y[i] = y[i-1] + rv[i] * np.sin(angle[i])
        if y[i] > bounds[1]:
            y[i] -= 2 * (y[i] - bounds[1])
        elif y[i] < -bounds[1]:
            y[i] -= 2 * (y[i] + bounds[1])

    points = np.array(list(zip(x,y)))

    # calculate elasped times
    times = np.zeros(2 * points.shape[0] - 1)
    times[0] = stay_time[0]

    for i in range(1, points.shape[0]):
        times[2*i-1] = times[2*(i-1)] + time_to_tread_path(points[i-1], points[i], K, RHO)
        times[2*i] = times[2*i-1] + stay_time[i]

    return points, times

def levy_walk(n, bounds, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta):
    K = 1.
    RHO = .5

    # uniformly distributed angles
    angle = uniform.rvs(size=(n,), loc=.0, scale=2.*np.pi)

    # levy distributed step length
    rv = truncated_levy(n, thres_loc, alpha)
    stay_time = truncated_levy(n, thres_time, beta)

    # x and y coordinates
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x_orig
    y[0] = y_orig

    # indices when it hit the wall. later used to interpolate points
    x_wall = set()
    y_wall = set()

    for i in range(1,n):
        x[i] = x[i-1] + rv[i] * np.cos(angle[i])
        if x[i] > bounds[0]:
            x_wall.add(i) 
            x[i] -= 2 * (x[i] - bounds[0])

        elif x[i] < -bounds[0]:
            x_wall.add(i) 
            x[i] -= 2 * (x[i] + bounds[0])

        y[i] = y[i-1] + rv[i] * np.sin(angle[i])
        if y[i] > bounds[1]:
            y_wall.add(i)
            y[i] -= 2 * (y[i] - bounds[1])

        elif y[i] < -bounds[1]:
            y_wall.add(i)
            y[i] -= 2 * (y[i] + bounds[1])
            
    points = np.array(list(zip(x,y)))

    # start constructing all the interpolations
    agg_points = np.empty([0,2])
    agg_times = np.empty([0,1])
    cur_t = 0

    for i in range(points.shape[0]-1):
        if i in x_wall or i in y_wall:
            if i in x_wall:
                hit_point = np.array([points[i][0], (points[i][1]+points[i][1])/2])
            elif i in y_wall:
                hit_point = np.array([(points[i][0]+points[i][0])/2, points[i][1]])

            # from i to hit point
            travel_time = time_to_tread_path(points[i], hit_point, K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(points[i], hit_point, interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)
            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size

            # from hit point to i
            travel_time = time_to_tread_path(hit_point, points[i+1], K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(hit_point, points[i+1], interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)
            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size
            
        else:
            travel_time = time_to_tread_path(points[i], points[i+1], K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(points[i], points[i+1], interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)

            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size

    return agg_points, agg_times

def levy_walk_episodes(n, bounds, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta, episodes, static_region=None):
    points, times = levy_walk(n, bounds, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta)

    for _ in range(episodes-1):
        p, t = levy_walk(n, bounds, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta)
        points = np.concatenate((points, p), axis=0)
        # t = np.insert(t, 0, times[-1])
        t += times[-1]
        times = np.concatenate((times, t), axis=0)

    # if a static region was specified, check if the starting point is in it and, if so, make the position static
    if static_region is not None:
        if x_orig >= -static_region[0] and x_orig <= static_region[0] and y_orig >= -static_region[1] and y_orig <= static_region[1]:
            points = [[x_orig, y_orig] for _ in points]

    return points, times