import os
import json
import numpy as np
import re
import tqdm
import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def sort_key(filename):
    """Sort function for filenames based on the numeric part."""
    return int(re.search(r'\d+', filename).group())


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_ne(path, dirs, success_dirs):
    """Calculate the NE (Normalized Error) between predicted and oracle trajectories."""
    ne_list = []
    
    for traj_dir in tqdm.tqdm(dirs, desc='Calculating NE'):
        log_dir = os.path.join(path, traj_dir, 'log')
        logs = sorted(os.listdir(log_dir), key=sort_key)

        last_log_data = load_json(os.path.join(log_dir, logs[-1]))
        last_point = np.array(last_log_data["sensors"]['state']['position'])

        ori_info = load_json(os.path.join(path, traj_dir, 'ori_info.json'))
        ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']
        ori_last_point = np.array(ori_data[-1]['position'])

        ne = np.linalg.norm(ori_last_point - last_point)
        ne_list.append(ne)

    avg_ne = np.mean(np.array(ne_list))
    logging.info(f"Average Normalized Error (NE): {avg_ne:.2f}")


def calculate_spl(path, dirs, success_dirs):
    """Calculate the SPL (Success Path Length) based on predicted and oracle trajectories."""
    spl_list = []

    for traj_dir in tqdm.tqdm(dirs, desc='Calculating SPL'):
        if traj_dir not in success_dirs:
            spl_list.append(0)
            continue

        log_dir = os.path.join(path, traj_dir, 'log')
        logs = sorted(os.listdir(log_dir), key=sort_key)

        pred_length = 0
        pre_point = None
        for log in logs:
            log_data = load_json(os.path.join(log_dir, log))
            point = np.array(log_data["sensors"]['state']['position'])
            if pre_point is not None:
                pred_length += np.linalg.norm(pre_point - point)
            pre_point = point

        ori_info = load_json(os.path.join(path, traj_dir, 'ori_info.json'))
        ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']

        path_length = 0
        for i in range(len(ori_data) - 1):
            p1 = np.array(ori_data[i]['position'])
            p2 = np.array(ori_data[i + 1]['position'])
            path_length += np.linalg.norm(p2 - p1)
        path_length -= 20  

        spl = path_length / max(path_length, pred_length)
        spl = max(spl, 0) 
        spl_list.append(spl)

    avg_spl = np.mean(np.array(spl_list)) * 100
    logging.info(f"Average Success Path Length (SPL): {avg_spl:.2f}%")


def split_data(path, path_type, map_filter=None):
    """Split the dataset into different categories based on the path type."""
    dirs = os.listdir(path)
    return_dirs = []

    if path_type == 'full':
        candidate_dirs = [traj_dir for traj_dir in dirs if 'record' not in traj_dir and 'dino' not in traj_dir]
    else:
        candidate_dirs = []
        for traj_dir in tqdm.tqdm(dirs, desc='Splitting data'):
            # Skip if ori_info.json doesn't exist
            ori_info_path = os.path.join(path, traj_dir, 'ori_info.json')
            if not os.path.exists(ori_info_path):
                continue
            
            ori_info = load_json(ori_info_path)
            ori_data = load_json(os.path.join(ori_info['ori_traj_dir'], 'merged_data.json'))['trajectory_raw_detailed']

            path_length = sum(np.linalg.norm(np.array(ori_data[i + 1]['position']) - np.array(ori_data[i]['position'])) for i in range(len(ori_data) - 1))

            if path_type == 'easy' and path_length <= 250:
                candidate_dirs.append((traj_dir, ori_info))
            elif path_type == 'hard' and path_length > 250:
                candidate_dirs.append((traj_dir, ori_info))
            elif 'unseen' in path_type:
                unseen_scenes = ['Carla_Town03', 'ModularPark']
                if path_type == 'unseen scene' and any(scene in ori_info['ori_traj_dir'] for scene in unseen_scenes):
                    candidate_dirs.append((traj_dir, ori_info))
                elif path_type == 'unseen object' and not any(scene in ori_info['ori_traj_dir'] for scene in unseen_scenes):
                    candidate_dirs.append((traj_dir, ori_info))
    
    # Apply map filter
    if map_filter is None:
        if path_type == 'full':
            return candidate_dirs
        else:
            return [traj_dir for traj_dir, _ in candidate_dirs]
    
    # Filter by map name
    filtered_dirs = []
    if path_type == 'full':
        for traj_dir in candidate_dirs:
            ori_info_path = os.path.join(path, traj_dir, 'ori_info.json')
            # Skip if ori_info.json doesn't exist
            if not os.path.exists(ori_info_path):
                continue
            ori_info = load_json(ori_info_path)
            if map_filter in ori_info['ori_traj_dir']:
                filtered_dirs.append(traj_dir)
    else:
        for traj_dir, ori_info in candidate_dirs:
            if map_filter in ori_info['ori_traj_dir']:
                filtered_dirs.append(traj_dir)
    
    return filtered_dirs



def analyze_results(root_dir, analysis_list, path_type_list, map_filter=None):
    """Main function to analyze the results for different analysis types and path types."""
    for analysis_item in analysis_list:
        analysis_path = os.path.join(root_dir, analysis_item)
        if not os.path.exists(analysis_path):
            continue
        
        map_info = f" (Map: {map_filter})" if map_filter else ""
        logging.info(f"\nStarting analysis for type: {analysis_item}{map_info}")

        for path_type in path_type_list:
            logging.info(f'\nAnalyzing for path type: {path_type}')
            analysis_dirs = split_data(analysis_path, path_type, map_filter)

            total = len(analysis_dirs)

            logging.info(f"Total Count: {total}")
            
            success = 0
            oracle = 0
            success_dirs = []

            for traj_dir in analysis_dirs:
                if 'success' in traj_dir:
                    success += 1
                    oracle += 1
                    success_dirs.append(traj_dir)
                elif 'oracle' in traj_dir:
                    oracle += 1

            sr = success / (total + 1e-8) * 100
            osr = oracle / (total + 1e-8) * 100
            logging.info(f"Success Rate (SR): {sr:.2f}%")
            logging.info(f"Oracle Success Rate (OSR): {osr:.2f}%")

            calculate_ne(analysis_path, analysis_dirs, success_dirs)
            calculate_spl(analysis_path, analysis_dirs, success_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the evaluation results for trajectory prediction.")
    parser.add_argument('--root_dir', type=str, required=True, help="The root directory of the dataset.")
    parser.add_argument('--analysis_list', type=str, nargs='+', required=True, help="List of analysis items to process.")
    parser.add_argument('--path_type_list', type=str, nargs='+', required=True, help="List of path types to analyze.")
    parser.add_argument('--map_filter', type=str, default=None, help="Filter results by map name (e.g., 'ModernCityMap', 'Carla_Town01')")

    args = parser.parse_args()

    analyze_results(args.root_dir, args.analysis_list, args.path_type_list, args.map_filter)

