import argparse
import threading
import traceback
import msgpackrpc
from pathlib import Path
import glob
import time
import os
import json
import sys
import subprocess
import errno
import signal
import copy


AIRSIM_SETTINGS_TEMPLATE = {
  "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 10,
  "ViewMode": "SpringArmChase",
  "PhysiceEngineName": "ExternalPhysicsEngine",
  "Recording": {
    "RecordInterval": 1,
    "Enabled": False,
    "Cameras": []
  },
  "Vehicles": {
    "Drone_1": {
      "VehicleType": "SimpleFlight",
      "UseSerial": False,
      "LockStep": True,
      "AutoCreate": True,
      "X": 0,
      "Y": 0,
      "Z": -2,
      "Roll": 0,
      "Pitch": 0,
      "Yaw": 0,
      "Cameras": {
        "FrontCamera": {
          "X": 1,
          "Y": 0,
          "Z": 0,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": 0,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "RearCamera": {
          "X": -1,
          "Y": 0,
          "Z": 0,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": 180,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "LeftCamera": {
          "X": 0,
          "Y": -1,
          "Z": 0,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": -90,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "RightCamera": {
          "X": 0,
          "Y": 1,
          "Z": 0,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": 90,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "DownCamera": {
          "X": 0,
          "Y": 0,
          "Z": 0,
          "Pitch": -90,
          "Roll": 0,
          "Yaw": 0,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 256,
              "Height": 256,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "FrontCameraRecord": {
          "X": 1,
          "Y": 0,
          "Z": 0,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": 0,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1024,
              "Height": 1024,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 1024,
              "Height": 1024,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        },
        "DownCameraRecord": {
          "X": 0,
          "Y": 0,
          "Z": 0,
          "Pitch": -90,
          "Roll": 0,
          "Yaw": 0,
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1024,
              "Height": 1024,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            },
            {
              "ImageType": 2,
              "Width": 1024,
              "Height": 1024,
              "FOV_Degrees": 90,
              "AutoExposureMaxBrightness": 1,
              "AutoExposureMinBrightness": 0.03
            }
          ]
        }
      },
      "Sensors": {
          "Imu": {
                "SensorType": 2,
                "Enabled" : True,
                "AngularRandomWalk": 0.3,
                "GyroBiasStabilityTau": 500,
                "GyroBiasStability": 4.6,
                "VelocityRandomWalk": 0.24,
                "AccelBiasStabilityTau": 800,
                "AccelBiasStability": 36
            }
      }
    }
  }
}

env_exec_path_dict = {
    "NYCEnvironmentMegapa": {
        'bash_name': 'NYCEnvironmentMegapa',
        'exec_path': './closeloop_envs',
    },
    "TropicalIsland": {
        'bash_name': 'TropicalIsland',
        'exec_path': './closeloop_envs',
    },
    "NewYorkCity": {
        'bash_name': 'NewYorkCity',
        'exec_path': './closeloop_envs',
    },
    "ModularPark": {
        'bash_name': 'ModularPark',
        'exec_path': './closeloop_envs',
    },
    "ModularEuropean": {
        'bash_name': 'ModularEuropean',
        'exec_path': './closeloop_envs',
    },
    "ModernCityMap": {
        'bash_name': 'ModernCityMap',
        'exec_path': './closeloop_envs',
    },
    "Carla_Town01": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town01/LinuxNoEditor',
    },
    "Carla_Town02": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town02/LinuxNoEditor',
    },
    "Carla_Town03": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town03/LinuxNoEditor',
    },
    "Carla_Town04": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town04/LinuxNoEditor',
    },
    "Carla_Town05": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town05/LinuxNoEditor',
    },
    "Carla_Town06": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town06/LinuxNoEditor',
    },
    "Carla_Town07": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town07/LinuxNoEditor',
    },
    "Carla_Town10HD": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town10HD/LinuxNoEditor',
    },
    "Carla_Town15": {
        'bash_name': 'CarlaUE4',
        'exec_path': './carla_town_envs/Town15/LinuxNoEditor',
    },
}
def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=True) -> dict:
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)
    return airsim_settings


def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def FromPortGetPid(port: int):
    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        return

    while pid_exists(pid):
        try:
            print('pid {} is killed'.format(pid))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port, args=(index, port), daemon=True)
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'KillAirVLN',
                e,
            )
        )
        return
    except:
        return

    try:
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return


class EventHandler(object):
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus

        self.scene_used_ports = []
        
        self.port_to_scene = {}

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str , scen_id_gpu_list: list):
        print(
            "{}\t关闭场景中".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        print(
            "{}\t已关闭所有场景".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_id_gpu_list):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        KillPorts(ports)

        # Occupied GPU 2
        gpus = [scen_id_gpu_list[index][-1] for index in range(len(scen_id_gpu_list))]
        print(scen_id_gpu_list)

        # search scene path 3
        choose_env_exe_paths = []
        for scen_id, gpu_id in scen_id_gpu_list:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue
            
            if scen_id in env_exec_path_dict:
                env_info = env_exec_path_dict.get(scen_id)
                res = os.path.join(args.root_path, env_info['exec_path'], env_info['bash_name'] + '.sh')
                choose_env_exe_paths.append(res)
            else:
                prefix_flag = False
                for map_name in env_exec_path_dict.keys():
                    if str(scen_id).startswith(map_name):
                        prefix_flag = True
                        env_info = env_exec_path_dict.get(map_name)
                        res = os.path.join(args.root_path, env_info['exec_path'], env_info['bash_name'] + '.sh')
                        choose_env_exe_paths.append(res)
                if not prefix_flag:
                    print(f'can not find scene file: {scen_id}')
                    raise KeyError

        p_s = []
        for index, (scen_id, gpu_id) in enumerate(scen_id_gpu_list):
            # airsim settings 4
            airsim_settings = create_drones()
            airsim_settings['ApiServerPort'] = int(ports[index])
            self.port_to_scene[ports[index]] = (scen_id, gpu_id)
            airsim_settings_write_content = json.dumps(airsim_settings)
            if not os.path.exists(str(CWD_DIR / 'settings' / str(ports[index]))):
                os.makedirs(str(CWD_DIR / 'settings' / str(ports[index])), exist_ok=True)
            with open(str(CWD_DIR / 'settings' / str(ports[index]) / 'settings.json'), 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)

            # open scene 5
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} -settings={} ".format(
                   choose_env_exe_paths[index],
                   gpu_id,
                   str(CWD_DIR / 'settings' / str(ports[index]) / 'settings.json'),
                )
               
                # subprocess_execute = "bash {} -windowed -ResX=1280 -ResY=720 -WinX=740 -WinY=610 -NoSound -NoVSync -GraphicsAdapter={} -settings={} ".format(
                #     choose_env_exe_paths[index],
                #     gpu_id,
                #     str(CWD_DIR / 'settings' / str(ports[index]) / 'settings.json'),
                # )
                
                time.sleep(1)
                print(subprocess_execute)

                try:
                    
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    p_s.append(p)
                except Exception as e:
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        time.sleep(10)
        self.scene_used_ports += copy.deepcopy(ports)
        
        print("finished", ip)

        return True, (ip, ports)
    
    def reopen_scene_from_port(self, port):

        KillPorts([port])
        
        scene_id, gpu_id = self.port_to_scene[port]
        env_info = env_exec_path_dict.get(scene_id)
        env_path = os.path.join(args.root_path, env_info['exec_path'], env_info['bash_name'] + '.sh')
        subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} -settings={} ".format(
                    env_path,
                    gpu_id,
                    str(CWD_DIR / 'settings' / str(port) / 'settings.json'),
                )
        time.sleep(1)
        print(subprocess_execute)
        
        p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                        shell=True,
                    )
        
    def reopen_scenes(self, ip: str, scen_id_gpu_list: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            print(scen_id_gpu_list)
            ip = ip
            # 
            for item in scen_id_gpu_list:
                if isinstance(item[0], bytes):
                    item[0] = item[0].decode('utf-8')
            #
            result = self._open_scenes(ip, scen_id_gpu_list)
        except Exception as e:
            print(e)
            exe_type, exe_value, exe_traceback = sys.exc_info()
            exe_info_list = traceback.format_exception(
                exe_type, exe_value, exe_traceback)
            tracebacks = ''.join(exe_info_list)
            print('traceback:', tracebacks)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread
    except Exception as err:
        print("error",err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help='server port'
    ) 
    parser.add_argument(
        "--root_path",
        type=str,
        default="../envs",
        help='root dir for env path'
    ) 
    args = parser.parse_args()


    HOST = '127.0.0.1'
    PORT = int(args.port)
    CWD_DIR = Path(str(os.path.abspath(__file__))).parent.resolve()
    PROJECT_ROOT_DIR = CWD_DIR.parent
    print("PROJECT_ROOT_DIR",PROJECT_ROOT_DIR)

    gpu_list = []
    gpus = str(args.gpus).split(',') 
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()


    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")

