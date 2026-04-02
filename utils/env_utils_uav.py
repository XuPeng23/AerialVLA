import math
import numba as nb
import airsim
import numpy as np
import copy

from src.common.param import args

from utils.logger import logger


class SimState:
    def __init__(self, index=-1,                                                                                                                                                        
                 step=0,
                 raw_trajectory_info={},
                 ):
        self.index = index
        self.step = step
        self.raw_trajectory_info = copy.deepcopy(raw_trajectory_info) # GT PATH
        self.trajectory = [{'sensor': {'state':self.raw_trajectory_info['trajectory'][0]}}] # PREDICT PATH
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.predict_start_index = 0
        self.history_start_indexes = [0]
        self.SUCCESS_DISTANCE = 20
        self.pre_carrot_idx = 0
        self.start_point_nearest_node_token = None
        self.end_point_nearest_node_token = None
        self.progress = 0.0
        self.waypoint = {}
        self.unique_path = None
        
    def get_gt_waypoints(self):
        gt_waypoints = []
        for info in self.raw_trajectory_info['trajectory']:
            gt_waypoints.append(info['position'][0:3])
        return gt_waypoints
    
    def revert2frames(self):
        assert len(self.trajectory) > 0
        if len(self.trajectory) > 10:
            self.step = self.step - 2
            self.trajectory = self.trajectory[0:-10]
            self.is_end = False
            self.is_collisioned = False
            self.history_start_indexes = self.history_start_indexes[0:-2]
            self.predict_start_index = self.history_start_indexes[-1]
        else:
            self.step = 0
            self.trajectory = self.trajectory[0:1]
            self.is_end = False
            self.is_collisioned = False
            self.history_start_indexes = [0]
            self.predict_start_index = 0
    
    @property
    def state(self): 
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self): # 
        return self.trajectory[-1]['sensors']['state']['position'] + self.trajectory[-1]['sensors']['state']['orientation']

class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state: SimState):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]
        oracle_success = state.oracle_success

        if args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
            teacher_action_path = get_waypoint_at(STEP_NUM = 7,DISTANCE=1, state=state)
            done = state.is_end
        elif args.run_type in ['eval'] and args.collect_type in ['dagger', 'SF']:
            teacher_action_path = None
            done = state.is_end
        else:
            logger.error('wrong type')
            raise NotImplementedError

        return (teacher_action_path, done, oracle_success), state

def get_waypoint_at(STEP_NUM: int,DISTANCE: int, state: SimState):
    
    raw_path = state.get_gt_waypoints().copy() # raw
    predict_xyz = state.pose[0:3]
    state_index = state.predict_start_index # update valid index
    
    x, y, z = predict_xyz[0], predict_xyz[1], predict_xyz[2]
    end_point = [0,0,0] 
    shortest_index = -1
    shortest_distance = 99999
    for idx in range(state_index, len(raw_path)):
        pos = raw_path[idx]
        cur_distance = math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2)
        if cur_distance < shortest_distance:
            shortest_index = idx
            shortest_distance = cur_distance
    state.predict_start_index = shortest_index # update valid index
    state.history_start_indexes.append(shortest_index)
    assert shortest_index != -1, "cannot find shortest_index,check path again"
    assert shortest_distance != 99999, "cannot find shortest_distance,check path again"
    
    end_point = raw_path[shortest_index]
    sub_path = []
    
    delta_unit = (np.array(end_point) - np.array(predict_xyz)) / (np.linalg.norm((np.array(end_point) - np.array(predict_xyz))) + 1e-8)
    
    while shortest_distance >= DISTANCE*1.5 :
        x = x + delta_unit[0] * DISTANCE
        y = y + delta_unit[1] * DISTANCE
        z = z + delta_unit[2] * DISTANCE
        sub_path.append([x,y,z])
        shortest_distance -= DISTANCE
        
    extend_raw_path = raw_path
    for i in range(STEP_NUM):
        extend_raw_path.append(raw_path[-1].copy())
    
    gt_append_index = shortest_index
    while len(sub_path) < STEP_NUM:
        sub_path.append(extend_raw_path[gt_append_index])
        gt_append_index += 1
        
    sub_path = sub_path[0:STEP_NUM]
    
    return sub_path
