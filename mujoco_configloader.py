import yaml
import numpy as np

try:
    with open("mujoco_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        common_config = config['COMMON'].copy()
        active_config = config[config['ACTIVE']].copy()
        common_config.update(active_config)
        config = common_config
except:
    print("Some problem with the YAML file.")
    import sys
    sys.exit(1)

config['vx_range'] = np.linspace(0.0, 1.0, num = config['V_VALUES'], dtype=np.float32)
if not config['training_is_border']: config['vx_range'] += config['VX_DELTA']
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])

if __name__=="__main__":
    if config['artificial_data']:
        from mujoco_datagen_artificial import artificial_datagen
        artificial_datagen()
    else:
        from mujoco_datagen_simulation import simulation_datagen
        simulation_datagen()
