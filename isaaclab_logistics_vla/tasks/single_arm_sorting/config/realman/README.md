# Realman 机器人配置说明

## USD场景文件

本配置使用 `simple_01.usd` 作为基础场景

## 机器人配置

配置需要引用USD文件中已有的机器人prim

### 机器人的Prim路径

请在 `__init__.py` 中的 `RealmanSingleArmSortingEnvCfg.__post_init__()` 方法中，将 `self.scene.robot` 的 `prim_path` 设置为USD文件中机器人的实际路径。

常见的路径格式：
- `{ENV_REGEX_NS}/BaseScene/Robot` - 如果机器人在BaseScene下
- `{ENV_REGEX_NS}/Robot` - 如果机器人在根路径下
- 其他路径根据USD文件实际结构调整

### 如何查找机器人路径

可以通过以下方式查找USD文件中机器人的路径：
1. 在Isaac Sim中打开USD文件
2. 查看Stage面板中的prim层级结构
3. 找到机器人（Realman）的prim路径
4. 在配置中使用该路径

## 注意事项

- 如果USD文件中已经包含地面和光源，可能需要移除配置中的 `plane` 和 `light`
- 确保机器人prim路径正确，否则会出现找不到机器人的错误
