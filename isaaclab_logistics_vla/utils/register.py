import collections
import importlib
import pkgutil
import inspect

class Registration():
    def __init__(self):
        self._tasks = collections.OrderedDict()
        self._entities = collections.OrderedDict()
        self._robots = collections.OrderedDict()
        self._conditions = collections.OrderedDict()
        self._config_managers = collections.OrderedDict()

        self._eeframe_configs = collections.OrderedDict()
        self._action_configs = collections.OrderedDict()

        self._env_configs = collections.OrderedDict()

    def auto_scan(self, package_path_or_name):
        """
        动态扫描指定包下的所有模块并导入
        """
        package = importlib.import_module(package_path_or_name)
        for loader, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            # 动态加载模块
            mod = importlib.import_module(modname)
    
    def add_task(self, task_name):
        def wrap(cls):
            self._tasks[task_name] = cls
            return cls
        return wrap
    
    def add_entity(self, entity_name):
        def wrap(cls):
            self._entities[entity_name] = cls
            return cls
        return wrap
    
    def add_robot(self, robot_name):
        def wrap(cls):
            self._robots[robot_name] = cls
            return cls
        return wrap

    def add_condition(self, condition_name):
        def wrap(cls):
            self._conditions[condition_name] = cls
            return cls
        return wrap
    
    def add_config_manager(self, config_manager_name):
        def wrap(cls):
            self._config_managers[config_manager_name] = cls
            return cls
        return wrap
    
    def add_eeframe_configs(self, eeframe_config):
        def wrap(cls):
            self._eeframe_configs[eeframe_config] = cls
            return cls
        return wrap
    
    def add_action_configs(self, action_config):
        def wrap(cls):
            self._action_configs[action_config] = cls
            return cls
        return wrap
    
    def add_env_configs(self, env_config):
        def wrap(cls):
            self._env_configs[env_config] = cls
            return cls
        return wrap
    
    def __getitem__(self, key):
        return self._tasks[key] or self._entities[key]
    
    def load_entity(self, key):
        return self._entities[key]
    
    def load_task(self, key):
        return self._tasks[key]
    
    def load_robot(self, key):
        return self._robots[key]
    
    def load_condition(self, key):
        return self._conditions[key]
    
    def load_config_manager(self, key):
        return self._config_managers[key]
    
    def load_eeframe_configs(self, key):
        return self._eeframe_configs[key]
    
    def load_action_configs(self, key):
        return self._action_configs[key]
    
    def load_env_configs(self, key):
        return self._env_configs[key]
    
    def keys(self):
        return self._tasks.keys()
    
    def __len__(self):
        return len(self._tasks)
    
    def __iter__(self):
        return iter(self._tasks)
    
    def get_robot_names(self):
        return self._robots.keys()
      
register = Registration()

