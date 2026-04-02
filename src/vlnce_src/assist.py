from src.vlnce_src.env_uav import RGB_FOLDER, DEPTH_FOLDER
from collections import deque


class Assist:

    def __init__(self, always_help = False, use_gt = False, device=0):
        self.always_help = always_help
        self.use_gt = use_gt
        self.recent_help_deque = deque(maxlen=9)

    def check_collision_by_depth(self, episodes, current_observations, collisions, dones):
        for i, prev_episode in enumerate(episodes):
            collision_type = None
            if collisions[i]:
                collision_type = 'already'
                if not dones[i]:
                    dones[i] = True
                continue
            
            # diffs = []
            close_collision = False
            current_episode = current_observations[i]
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                # diff = np.mean(np.abs(prev_episode[-1]['depth'][cid] - current_episode[-1]['depth'][cid]))
                zero_cnt  = (current_episode[-1]['depth'][cid] <= 1).sum()
                if zero_cnt > 0.1 * current_episode[-1]['depth'][cid].size:
                    close_collision = True
                    print(f"DEBUG: Camera {camera_name} trigger collision!")
                # diffs.append(diff)
            # distance = np.array(prev_episode[-1]["sensors"]["state"]["position"]) - np.array(current_episode[-1]["sensors"]["state"]["position"])
            # distance = np.linalg.norm(np.array(distance))
            # diffs = np.array(diffs)

            # Replaced pixel-diff based stuck detection with global displacement check (in closeloop_util.py) to support agile 3D maneuvers.
            
            # if np.all(diffs < 3):
                # collision_type = 'tiny diff'
            if close_collision:
                collision_type = 'close'
            # if distance < 0.1:
            #     collision_type = 'distance'
            
            if collision_type is not None:
                print('★★★★★collision type: ', collision_type)

            # collisions[i] = np.all(diff < 3) or close_collision or distance < 0.1
            collisions[i] = close_collision

            if collisions[i] and not dones[i]:
                dones[i] = True

        return collisions, dones

if __name__ == '__main__':
    ass = Assist()
