import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import shutil
import atexit
import signal

from run import REGISTRY as run_REGISTRY

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")

# Global variables to track directories that need to be cleaned up
_cleanup_dirs = []  # 存储需要清理的目录路径
_training_completed = False  # 标记训练是否正常完成
_results_base = join(dirname(dirname(abspath(__file__))), "results")

def add_cleanup_path(path):
    """添加需要在异常退出时清理的路径"""
    global _cleanup_dirs
    if path and path not in _cleanup_dirs:
        _cleanup_dirs.append(path)
        print(f"[Cleanup] 已注册清理路径: {path}")

def cleanup_failed_training():
    """清理训练失败时产生的所有文件"""
    global _cleanup_dirs, _training_completed
    
    if _training_completed:
        print("[Cleanup] 训练正常完成，跳过清理")
        return
    
    if not _cleanup_dirs:
        print("[Cleanup] 没有需要清理的目录")
        return
    
    print("\n" + "="*70)
    print("⚠️  检测到异常终止，开始清理运行期间产生的文件...")
    print("="*70)
    
    cleaned_count = 0
    failed_count = 0
    
    for dir_path in _cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"✓ 已删除: {dir_path}")
                cleaned_count += 1
            except Exception as e:
                print(f"✗ 删除失败: {dir_path}")
                print(f"  错误信息: {e}")
                failed_count += 1
        else:
            print(f"ℹ️  路径不存在，跳过: {dir_path}")
    
    print("="*70)
    if cleaned_count > 0:
        print(f"✓ 成功清理 {cleaned_count} 个目录")
    if failed_count > 0:
        print(f"✗ {failed_count} 个目录清理失败")
    print("="*70 + "\n")

def signal_handler(sig, frame):
    """处理中断信号（Ctrl+C 和终止信号）"""
    global _training_completed
    signal_name = "SIGINT (Ctrl+C)" if sig == signal.SIGINT else "SIGTERM"
    print(f"\n\n🛑 接收到 {signal_name} 信号，程序异常终止...")
    _training_completed = False  # 确保会执行清理
    cleanup_failed_training()
    sys.exit(1)

@ex.main
def my_main(_run, _config, _log):
    global _training_completed
    
    # 注册 tb_logs 清理路径（在运行开始时就注册可能的路径）
    if _config.get('use_tensorboard', False):
        # 构建 tb_logs 的基础路径
        tb_base = join(_results_base, "tb_logs")
        
        # 根据配置确定子目录
        if _config.get('training_attack'):
            tb_base = join(tb_base, "train_attack")
        elif _config.get('test_attack'):
            tb_base = join(tb_base, "test_attack")
        elif _config.get('all_test_attack'):
            tb_base = join(tb_base, "all_test_attack")
        else:
            tb_base = join(tb_base, "no_attack")
        
        # 添加环境相关路径
        if "sc2" in _config.get('env', '') or "gfootball" in _config.get('env', ''):
            tb_base = join(tb_base, _config['env_args'].get("map_name", ""))
        elif "mpe" in _config.get('env', ''):
            tb_base = join(tb_base, _config['env_args'].get("key", ""), 
                          str(_config['env_args'].get('num_agents', '')) + '_agents')
        elif "stag_hunt" in _config.get('env', ''):
            tb_base = join(tb_base, _config.get('env', ''), 
                          str(_config['env_args'].get('n_agents', '')) + '_agents')
        else:
            tb_base = join(tb_base, _config.get('env', ''))
        
        # 添加 epsilon 路径
        if 'epsilon' in _config:
            tb_base = join(tb_base, f"epsilon_{_config['epsilon']}")
        
        # 添加 pareto/robust_regular/diff_regular 路径
        if _config.get('pareto'):
            tb_base = join(tb_base, "pareto")
        elif _config.get('robust_regular'):
            tb_base = join(tb_base, "robust_regular", str(_config.get('robust_lambda', '')))
        elif _config.get('diff_regular'):
            tb_base = join(tb_base, "diff_regular", str(_config.get('robust_lambda', '')))
        
        # 注册 tb_logs 目录（会在运行时创建包含时间戳的子目录）
        add_cleanup_path(tb_base)
    
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    try:
        # run
        if "use_per" in _config and _config["use_per"]:
            run_REGISTRY['per_run'](_run, config, _log)
        else:
            run_REGISTRY[_config['run']](_run, config, _log)
        
        # 如果执行到这里，说明训练正常完成
        _training_completed = True
        print("[Cleanup] 训练正常完成，不会删除结果文件")
    except KeyboardInterrupt:
        print("\n[Cleanup] 用户中断训练")
        raise
    except Exception as e:
        logger.console_logger.error(f"[Cleanup] 训练失败: {e}")
        raise

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    # Register signal handlers FIRST, before anything else
    # This ensures Ctrl+C and termination signals are caught
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle kill command
    
    params = deepcopy(sys.argv)

    print(sys.argv[1:])
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['t_max'] = env_config['t_max']

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    # 注册 Sacred 结果目录用于清理
    # Sacred 会在这个目录下创建编号子目录（1, 2, 3...）
    add_cleanup_path(file_obs_path)
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    
    # 注册退出时的清理函数
    atexit.register(cleanup_failed_training)

    try:
        ex.run_commandline(params)
        # 正常完成
        _training_completed = True
        print("[Cleanup] 实验正常完成")
    except KeyboardInterrupt:
        print("\n[Cleanup] 用户中断实验")
    except Exception as e:
        print(f"\n[Cleanup] 实验执行失败: {e}")
        import traceback
        traceback.print_exc()

    # flush
    sys.stdout.flush()
