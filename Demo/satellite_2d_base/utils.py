import os
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    读取 YAML 配置文件并返回字典；若文件不存在则返回空字典。
    仅为满足上层模块的依赖，当前项目不强制使用该配置。
    """
    try:
        import yaml  # 轻量依赖
    except Exception:
        # 若环境未安装 PyYAML，则返回空配置，避免阻塞加载流程
        return {}

    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


