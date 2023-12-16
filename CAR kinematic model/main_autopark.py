import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger


# Data port for detection result
# bound_marking_points, car_points, wall_endpoints should be np array, and x_start, y_start are integers
# 1. car_points: dictionary, {1: [[41, 80]], 2: [[47, 80]], 3: [[53, 80]],..., n: [[XX, XX]]]
# 2. empty_index: list, key of empty carport
# 3. wall_endpoints: np array, [[[x_start1, y_start1],[x_end1, y_end1]], [[x_start2, y_start2],[x_end2, y_end2]], ... ], the wall should be horizontal ot vertical
# 4. start_point: np array, [x_start, y_start] of the edo vehicle's initial state
# 5. figure_size: the width and height of the hole figure, np array, [length, width]
# 6. car_size: the width and height of the car, np array, [width, height]

def set_up_map(car_points, empty_index, wall_endpoints, start_point, fig_size, car_size):
    # defining obstacles
    parking1 = Parking1(start_point[0],start_point[1])
    parking1.cars = car_points
    for wall in wall_endpoints:
        if wall[0,0] == wall[1,0]:
            if parking1.walls == None:
                parking1.walls = [[ wall[0,0], i] for i in range(wall[0,1], wall[1,1])]
            else:
                parking1.walls = parking1.walls+[[ wall[0,0], i] for i in range(wall[0,1], wall[1,1])]
        if wall[0,1] == wall[1,1]:
            if parking1.walls == None:
                parking1.walls = [[ i, wall[0,1]] for i in range(wall[0,0], wall[1,0])]
            else:
                parking1.walls = parking1.walls+[[ i, wall[0,1]] for i in range(wall[0,0], wall[1,0])] 
    parking1.obs = np.array(parking1.walls)
    parking1.empty_pos = empty_index
    costs = []
    for i in empty_index:
        cost = abs(parking1.x-parking1.cars[i][0][0]) + abs(parking1.y-parking1.cars[i][0][1])
        costs.append(cost)
    car_pos = parking1.empty_pos[costs.index(min(costs))]
    parking1.end = parking1.cars[car_pos][0]
    parking1.cars.pop(car_pos)    
    end, obs = parking1.generate_obstacles()
    
    # setup environment
    env = Environment(obs, width=fig_size[0], height=fig_size[1], car_length = car_size[0], car_width = car_size[1])
    
    return obs, env, end
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    
    ########################### initialization ##################################################
    
    # Uncomment for inputing detection data
    # obs, env, end = set_up_map(car_points, wall_endpoints, start_point, fig_size, car_size)
    
    # Comment the following 3 lines for inputing detection data
    parking1 = Parking1(args.x_start,args.y_start)
    end, obs = parking1.generate_obstacles()
    env = Environment(obs)
    
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controllerc = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    print('planning park scenario ...')
    new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    
    print('routing to destination ...')

    path = path_planner.plan_path(int(new_end[0]),int(new_end[1]),int(start[0]),int(start[1]))
    path = np.vstack([path[::-1], ensure_path1])

    print('interpolating ...')
    interpolated_path = interpolate_path(path, sample_rate=5)
    interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

    env.draw_path(interpolated_path)
    env.draw_path(interpolated_park_path)

    final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    #############################################################################################

    ################################## control ##################################################
    print('driving to destination ...')
    for i,point in enumerate(final_path):
        
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            my_car.update_state(my_car.move(acc,  delta))
            res = env.render(my_car.x, my_car.y, my_car.psi, delta)
            logger.log(point, my_car, acc, delta)
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()

