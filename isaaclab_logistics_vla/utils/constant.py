import os
ASSET_ROOT_PATH = os.getenv("ASSET_ROOT_PATH", "")

WORK_BOX_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/env/Box.usd",
    'X_LENGTH':0.36,
    'Y_LENGTH' :0.56,
    'Z_LENGTH':0.23
}

#---障碍物参数(新增)---
LARGE_OBSTACLE_PARAMS = {
    'X_LENGTH': 0.30,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.30,
    'COLOR': (0.8, 0.1, 0.1)
}

tray_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/env/Collected_Blue_Tray/SM_Crate_A08_Blue_01.usd",
    'X_LENGTH':1.2,
    'Y_LENGTH':0.76,
    'Z_LENGTH':0.32,
    #SPARSE_ORIENT':(0,0,0)    #相对于箱子的坐标
}

CRACKER_BOX_PARAMS = {
    'NAME': "cracker_box",
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_003_cracker_box/003_cracker_box.usd",
    'X_LENGTH':0.16,
    'Y_LENGTH':0.20,
    'Z_LENGTH':0.06,
    'SPARSE_ORIENT':(0,90,0),   #相对于箱子的坐标
    "DENSE_ORIENT":[(0,90,0),(0,0,0)],
    'STACK_ORIENT':(0,0,0),     # Z最小，默认朝向即可
    'STACK_SCALE': 0.6
}

SUGER_BOX_PARAMS = {
    'NAME': "sugar_box",
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_004_sugar_box/004_sugar_box.usd",
    'X_LENGTH':0.09,
    'Y_LENGTH':0.17,
    'Z_LENGTH':0.04,
    'SPARSE_ORIENT':(0,90,0),
    "DENSE_ORIENT":[(0,90,0),(0,0,0)],
    'STACK_ORIENT':(0,0,0),        # Z最小，默认朝向即可
    'STACK_SCALE': 0.6,
}

TOMATO_SOUP_CAN_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_005_tomato_soup_can/005_tomato_soup_can.usd",
    'RADIUS':0.035,
    'X_LENGTH':0.07,
    'Y_LENGTH':0.10,
    'Z_LENGTH':0.07,
    'SPARSE_ORIENT':(90,0,0),
    "DENSE_ORIENT":[(90,0,0),(0,0,0)],
     'STACK_ORIENT':(0,0,0)         # X=Z 等大，默认朝向即可
}

CN_BIG_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_CNBig/CNBig.usdc",
    "RADIUS":0.275,
    'X_LENGTH':0.55,
    'Y_LENGTH':0.14,
    'Z_LENGTH':0.14,
    'SPARSE_ORIENT':(0,0,90),#or (0,0,0)
    "DENSE_ORIENT":[(0,0,90),(0,0,0)]
}

SF_SMALL_PARAMS = {
    'NAME': "sf_small",
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_SFSmall/SFSmall.usdc",
    "RADIUS":0.035,
    'X_LENGTH':0.34,    
    'Y_LENGTH':0.43,
    'Z_LENGTH':0.08,
    'SPARSE_ORIENT':(0,0,0),#or (0,0,0)
    'STACK_ORIENT':(0,0,0),
    "DENSE_ORIENT":[(0,0,0)],
    'STACK_SCALE': 0.3,
}

EMPTY_PLASTIC_PACKAGE_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_empty_plastic_package/empty_plastic_package.usdc",
    "RADIUS":0.185,
    'X_LENGTH':0.35,    
    'Y_LENGTH':0.37,
    'Z_LENGTH':0.07,
    'SPARSE_ORIENT':(0,0,0),#or (0,0,0)
    "DENSE_ORIENT":[(0,0,0)]
}

SF_BIG_PARAMS = {
    'NAME': "sf_big",
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_SFBig/SFBig.usdc",
    "RADIUS":0.22,
    'X_LENGTH':0.47,    
    'Z_LENGTH':0.34,
    'Y_LENGTH':0.15,
    'SPARSE_ORIENT':(90,0,0), # (0,0,0)
    'STACK_ORIENT':(90,0,90),
    "DENSE_ORIENT":[(90,0,0),(0,0,0)],
    'STACK_SCALE': 0.3,
}

PLASTIC_PACKAGE_PARAMS = {
    'NAME': "plastic_package",
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_plastic_package/plastic_package.usdc",
    'X_LENGTH':0.34,
    'Y_LENGTH':0.39,
    'Z_LENGTH':0.07,
    'SPARSE_ORIENT':(0,0,0),
    'STACK_ORIENT':(0,0,0),        # Z最小，默认朝向即可
    'STACK_SCALE': 0.4,
}

# NOTE: 该文件由脚本自动生成（包含手写常量前缀 + CSV 物品区），请勿手工修改 CSV 生成区。
# 生成脚本：isaaclab_logistics_vla/scripts/generate_constants_from_robotwin_csv.py

BOTTLE_BASE1 = {
    'NAME': 'bottle_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/114_bottle/visual/base1.usd",
    'X_LENGTH': 0.058,
    'Y_LENGTH': 0.058,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOTTLE_BASE2 = {
    'NAME': 'bottle_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/114_bottle/visual/base2.usd",
    'X_LENGTH': 0.051,
    'Y_LENGTH': 0.051,
    'Z_LENGTH': 0.21,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOTTLE_BASE3 = {
    'NAME': 'bottle_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/114_bottle/visual/base3.usd",
    'X_LENGTH': 0.052,
    'Y_LENGTH': 0.188,
    'Z_LENGTH': 0.051,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOTTLE_BASE4 = {
    'NAME': 'bottle_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/114_bottle/visual/base4.usd",
    'X_LENGTH': 0.051,
    'Y_LENGTH': 0.051,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOXDRINK_BASE0 = {
    'NAME': 'boxdrink_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/068_boxdrink/visual/base0.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.115,
    'Z_LENGTH': 0.154,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOXDRINK_BASE1 = {
    'NAME': 'boxdrink_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/068_boxdrink/visual/base1.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.254,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOXDRINK_BASE2 = {
    'NAME': 'boxdrink_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/068_boxdrink/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.306,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOXDRINK_BASE3 = {
    'NAME': 'boxdrink_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/068_boxdrink/visual/base3.usd",
    'X_LENGTH': 0.079,
    'Y_LENGTH': 0.062,
    'Z_LENGTH': 0.254,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE0 = {
    'NAME': 'can_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base0.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.175,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE1 = {
    'NAME': 'can_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE2 = {
    'NAME': 'can_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.192,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE3 = {
    'NAME': 'can_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base3.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE5 = {
    'NAME': 'can_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base5.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.192,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CAN_BASE6 = {
    'NAME': 'can_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/071_can/visual/base6.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE0 = {
    'NAME': 'coffeebox_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base0.usd",
    'X_LENGTH': 0.101,
    'Y_LENGTH': 0.091,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE1 = {
    'NAME': 'coffeebox_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base1.usd",
    'X_LENGTH': 0.131,
    'Y_LENGTH': 0.121,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE2 = {
    'NAME': 'coffeebox_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base2.usd",
    'X_LENGTH': 0.131,
    'Y_LENGTH': 0.111,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE3 = {
    'NAME': 'coffeebox_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base3.usd",
    'X_LENGTH': 0.101,
    'Y_LENGTH': 0.091,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE4 = {
    'NAME': 'coffeebox_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base4.usd",
    'X_LENGTH': 0.131,
    'Y_LENGTH': 0.101,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE5 = {
    'NAME': 'coffeebox_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base5.usd",
    'X_LENGTH': 0.151,
    'Y_LENGTH': 0.031,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

COFFEEBOX_BASE6 = {
    'NAME': 'coffeebox_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/113_coffeebox/visual/base6.usd",
    'X_LENGTH': 0.091,
    'Y_LENGTH': 0.101,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE0 = {
    'NAME': 'fan_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base0.usd",
    'X_LENGTH': 0.104,
    'Y_LENGTH': 0.088,
    'Z_LENGTH': 0.16,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE1 = {
    'NAME': 'fan_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base1.usd",
    'X_LENGTH': 0.135,
    'Y_LENGTH': 0.118,
    'Z_LENGTH': 0.155,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE2 = {
    'NAME': 'fan_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base2.usd",
    'X_LENGTH': 0.135,
    'Y_LENGTH': 0.115,
    'Z_LENGTH': 0.16,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE3 = {
    'NAME': 'fan_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base3.usd",
    'X_LENGTH': 0.128,
    'Y_LENGTH': 0.102,
    'Z_LENGTH': 0.155,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE4 = {
    'NAME': 'fan_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base4.usd",
    'X_LENGTH': 0.112,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.152,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE5 = {
    'NAME': 'fan_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base5.usd",
    'X_LENGTH': 0.128,
    'Y_LENGTH': 0.104,
    'Z_LENGTH': 0.16,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FAN_BASE6 = {
    'NAME': 'fan_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/099_fan/visual/base6.usd",
    'X_LENGTH': 0.136,
    'Y_LENGTH': 0.096,
    'Z_LENGTH': 0.154,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE0 = {
    'NAME': 'glue_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base0.usd",
    'X_LENGTH': 0.065,
    'Y_LENGTH': 0.065,
    'Z_LENGTH': 0.194,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE1 = {
    'NAME': 'glue_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base1.usd",
    'X_LENGTH': 0.065,
    'Y_LENGTH': 0.065,
    'Z_LENGTH': 0.192,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE2 = {
    'NAME': 'glue_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base2.usd",
    'X_LENGTH': 0.068,
    'Y_LENGTH': 0.068,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE4 = {
    'NAME': 'glue_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base4.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.192,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE5 = {
    'NAME': 'glue_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base5.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.063,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

GLUE_BASE6 = {
    'NAME': 'glue_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/095_glue/visual/base6.usd",
    'X_LENGTH': 0.054,
    'Y_LENGTH': 0.054,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE0 = {
    'NAME': 'kettle_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base0.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE1 = {
    'NAME': 'kettle_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base1.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE2 = {
    'NAME': 'kettle_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base2.usd",
    'X_LENGTH': 0.14,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE3 = {
    'NAME': 'kettle_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base3.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE4 = {
    'NAME': 'kettle_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base4.usd",
    'X_LENGTH': 0.173,
    'Y_LENGTH': 0.175,
    'Z_LENGTH': 0.196,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE5 = {
    'NAME': 'kettle_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/091_kettle/visual/base5.usd",
    'X_LENGTH': 0.14,
    'Y_LENGTH': 0.2,
    'Z_LENGTH': 0.17,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KEYBOARD_BASE0 = {
    'NAME': 'keyboard_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/116_keyboard/visual/base0.usd",
    'X_LENGTH': 0.194,
    'Y_LENGTH': 0.068,
    'Z_LENGTH': 0.008,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KEYBOARD_BASE1 = {
    'NAME': 'keyboard_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/116_keyboard/visual/base1.usd",
    'X_LENGTH': 0.194,
    'Y_LENGTH': 0.068,
    'Z_LENGTH': 0.028,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KEYBOARD_BASE2 = {
    'NAME': 'keyboard_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/116_keyboard/visual/base2.usd",
    'X_LENGTH': 0.194,
    'Y_LENGTH': 0.068,
    'Z_LENGTH': 0.028,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KEYBOARD_BASE3 = {
    'NAME': 'keyboard_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/116_keyboard/visual/base3.usd",
    'X_LENGTH': 0.194,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.028,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE0 = {
    'NAME': 'milktea_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base0.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.12,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE1 = {
    'NAME': 'milktea_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.206,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE2 = {
    'NAME': 'milktea_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base2.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE4 = {
    'NAME': 'milktea_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base4.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.212,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE5 = {
    'NAME': 'milktea_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base5.usd",
    'X_LENGTH': 0.119,
    'Y_LENGTH': 0.121,
    'Z_LENGTH': 0.197,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILKTEA_BASE6 = {
    'NAME': 'milktea_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/101_milktea/visual/base6.usd",
    'X_LENGTH': 0.109,
    'Y_LENGTH': 0.109,
    'Z_LENGTH': 0.191,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE0 = {
    'NAME': 'msg_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base0.usd",
    'X_LENGTH': 0.088,
    'Y_LENGTH': 0.088,
    'Z_LENGTH': 0.194,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE1 = {
    'NAME': 'msg_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base1.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.12,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE2 = {
    'NAME': 'msg_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base2.usd",
    'X_LENGTH': 0.078,
    'Y_LENGTH': 0.078,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE3 = {
    'NAME': 'msg_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base3.usd",
    'X_LENGTH': 0.078,
    'Y_LENGTH': 0.078,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE4 = {
    'NAME': 'msg_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base4.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MSG_BASE5 = {
    'NAME': 'msg_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/064_msg/visual/base5.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

NOTEBOOK_BASE0 = {
    'NAME': 'notebook_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/092_notebook/visual/base0.usd",
    'X_LENGTH': 0.191,
    'Y_LENGTH': 0.125,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

NOTEBOOK_BASE1 = {
    'NAME': 'notebook_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/092_notebook/visual/base1.usd",
    'X_LENGTH': 0.192,
    'Y_LENGTH': 0.129,
    'Z_LENGTH': 0.022,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

NOTEBOOK_BASE2 = {
    'NAME': 'notebook_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/092_notebook/visual/base2.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.129,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PERFUME_BASE0 = {
    'NAME': 'perfume_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/115_perfume/visual/base0.usd",
    'X_LENGTH': 0.135,
    'Y_LENGTH': 0.057,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PERFUME_BASE1 = {
    'NAME': 'perfume_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/115_perfume/visual/base1.usd",
    'X_LENGTH': 0.095,
    'Y_LENGTH': 0.075,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PERFUME_BASE2 = {
    'NAME': 'perfume_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/115_perfume/visual/base2.usd",
    'X_LENGTH': 0.103,
    'Y_LENGTH': 0.062,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PERFUME_BASE3 = {
    'NAME': 'perfume_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/115_perfume/visual/base3.usd",
    'X_LENGTH': 0.122,
    'Y_LENGTH': 0.111,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PHONE_BASE0 = {
    'NAME': 'phone_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/077_phone/visual/base0.usd",
    'X_LENGTH': 0.096,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.6,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PHONE_BASE1 = {
    'NAME': 'phone_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/077_phone/visual/base1.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.082,
    'Z_LENGTH': 0.018,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 90),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.6,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PHONE_BASE2 = {
    'NAME': 'phone_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/077_phone/visual/base2.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.095,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 90),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.6,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PHONE_BASE3 = {
    'NAME': 'phone_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/077_phone/visual/base3.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.098,
    'Z_LENGTH': 0.024,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 90),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.6,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PHONE_BASE4 = {
    'NAME': 'phone_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/077_phone/visual/base4.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.099,
    'Z_LENGTH': 0.023,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 90),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PILLBOTTLE_BASE1 = {
    'NAME': 'pillbottle_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/080_pillbottle/visual/base1.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.063,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PILLBOTTLE_BASE2 = {
    'NAME': 'pillbottle_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/080_pillbottle/visual/base2.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.063,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PILLBOTTLE_BASE3 = {
    'NAME': 'pillbottle_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/080_pillbottle/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PILLBOTTLE_BASE4 = {
    'NAME': 'pillbottle_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/080_pillbottle/visual/base4.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.063,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PILLBOTTLE_BASE5 = {
    'NAME': 'pillbottle_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/080_pillbottle/visual/base5.usd",
    'X_LENGTH': 0.055,
    'Y_LENGTH': 0.055,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE0 = {
    'NAME': 'plasticbox_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base0.usd",
    'X_LENGTH': 0.21,
    'Y_LENGTH': 0.161,
    'Z_LENGTH': 0.075,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE1 = {
    'NAME': 'plasticbox_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base1.usd",
    'X_LENGTH': 0.195,
    'Y_LENGTH': 0.195,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE10 = {
    'NAME': 'plasticbox_base10',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base10.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.16,
    'Z_LENGTH': 0.087,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE2 = {
    'NAME': 'plasticbox_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base2.usd",
    'X_LENGTH': 0.23,
    'Y_LENGTH': 0.23,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE3 = {
    'NAME': 'plasticbox_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base3.usd",
    'X_LENGTH': 0.21,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.075,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE4 = {
    'NAME': 'plasticbox_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base4.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.17,
    'Z_LENGTH': 0.094,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE5 = {
    'NAME': 'plasticbox_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base5.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.16,
    'Z_LENGTH': 0.076,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE6 = {
    'NAME': 'plasticbox_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base6.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.082,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE7 = {
    'NAME': 'plasticbox_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base7.usd",
    'X_LENGTH': 0.23,
    'Y_LENGTH': 0.13,
    'Z_LENGTH': 0.074,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE8 = {
    'NAME': 'plasticbox_base8',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base8.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.17,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLASTICBOX_BASE9 = {
    'NAME': 'plasticbox_base9',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/062_plasticbox/visual/base9.usd",
    'X_LENGTH': 0.22,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.092,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLAYINGCARDS_BASE0 = {
    'NAME': 'playingcards_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/081_playingcards/visual/base0.usd",
    'X_LENGTH': 0.072,
    'Y_LENGTH': 0.114,
    'Z_LENGTH': 0.024,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLAYINGCARDS_BASE1 = {
    'NAME': 'playingcards_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/081_playingcards/visual/base1.usd",
    'X_LENGTH': 0.115,
    'Y_LENGTH': 0.078,
    'Z_LENGTH': 0.024,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

PLAYINGCARDS_BASE2 = {
    'NAME': 'playingcards_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/081_playingcards/visual/base2.usd",
    'X_LENGTH': 0.114,
    'Y_LENGTH': 0.076,
    'Z_LENGTH': 0.024,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE0 = {
    'NAME': 'remotecontrol_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base0.usd",
    'X_LENGTH': 0.194,
    'Y_LENGTH': 0.048,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE1 = {
    'NAME': 'remotecontrol_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base1.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.057,
    'Z_LENGTH': 0.022,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE2 = {
    'NAME': 'remotecontrol_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base2.usd",
    'X_LENGTH': 0.065,
    'Y_LENGTH': 0.194,
    'Z_LENGTH': 0.039,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE3 = {
    'NAME': 'remotecontrol_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.024,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE4 = {
    'NAME': 'remotecontrol_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base4.usd",
    'X_LENGTH': 0.068,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.026,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE5 = {
    'NAME': 'remotecontrol_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base5.usd",
    'X_LENGTH': 0.062,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.031,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REMOTECONTROL_BASE6 = {
    'NAME': 'remotecontrol_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/079_remotecontrol/visual/base6.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.039,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REST_BASE0 = {
    'NAME': 'rest_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/094_rest/visual/base0.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.024,
    'Z_LENGTH': 0.089,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REST_BASE1 = {
    'NAME': 'rest_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/094_rest/visual/base1.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.071,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REST_BASE2 = {
    'NAME': 'rest_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/094_rest/visual/base2.usd",
    'X_LENGTH': 0.192,
    'Y_LENGTH': 0.062,
    'Z_LENGTH': 0.091,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

REST_BASE3 = {
    'NAME': 'rest_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/094_rest/visual/base3.usd",
    'X_LENGTH': 0.071,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.106,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ROLLER_BASE0 = {
    'NAME': 'roller_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/102_roller/visual/base0.usd",
    'X_LENGTH': 0.035,
    'Y_LENGTH': 0.193,
    'Z_LENGTH': 0.034,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ROLLER_BASE1 = {
    'NAME': 'roller_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/102_roller/visual/base1.usd",
    'X_LENGTH': 0.041,
    'Y_LENGTH': 0.191,
    'Z_LENGTH': 0.035,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ROLLER_BASE2 = {
    'NAME': 'roller_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/102_roller/visual/base2.usd",
    'X_LENGTH': 0.193,
    'Y_LENGTH': 0.032,
    'Z_LENGTH': 0.031,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

RUBIKSCUBE_BASE0 = {
    'NAME': 'rubikscube_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/073_rubikscube/visual/base0.usd",
    'X_LENGTH': 0.097,
    'Y_LENGTH': 0.115,
    'Z_LENGTH': 0.101,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.8,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

RUBIKSCUBE_BASE1 = {
    'NAME': 'rubikscube_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/073_rubikscube/visual/base1.usd",
    'X_LENGTH': 0.116,
    'Y_LENGTH': 0.116,
    'Z_LENGTH': 0.116,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.8,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

RUBIKSCUBE_BASE2 = {
    'NAME': 'rubikscube_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/073_rubikscube/visual/base2.usd",
    'X_LENGTH': 0.116,
    'Y_LENGTH': 0.116,
    'Z_LENGTH': 0.116,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 0.8,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SAUCECAN_BASE0 = {
    'NAME': 'saucecan_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/105_saucecan/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SAUCECAN_BASE2 = {
    'NAME': 'saucecan_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/105_saucecan/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SAUCECAN_BASE4 = {
    'NAME': 'saucecan_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/105_saucecan/visual/base4.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SAUCECAN_BASE5 = {
    'NAME': 'saucecan_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/105_saucecan/visual/base5.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SAUCECAN_BASE6 = {
    'NAME': 'saucecan_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/105_saucecan/visual/base6.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SMALLSPEAKER_BASE1 = {
    'NAME': 'smallspeaker_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/055_smallspeaker/visual/base1.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SMALLSPEAKER_BASE2 = {
    'NAME': 'smallspeaker_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/055_smallspeaker/visual/base2.usd",
    'X_LENGTH': 0.112,
    'Y_LENGTH': 0.112,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SMALLSPEAKER_BASE3 = {
    'NAME': 'smallspeaker_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/055_smallspeaker/visual/base3.usd",
    'X_LENGTH': 0.128,
    'Y_LENGTH': 0.064,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOAP_BASE0 = {
    'NAME': 'soap_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/107_soap/visual/base0.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.062,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOAP_BASE1 = {
    'NAME': 'soap_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/107_soap/visual/base1.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.062,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOAP_BASE2 = {
    'NAME': 'soap_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/107_soap/visual/base2.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.092,
    'Z_LENGTH': 0.078,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOAP_BASE3 = {
    'NAME': 'soap_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/107_soap/visual/base3.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.072,
    'Z_LENGTH': 0.073,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOYSAUCE_BASE0 = {
    'NAME': 'soysauce_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/065_soysauce/visual/base0.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.255,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOYSAUCE_BASE1 = {
    'NAME': 'soysauce_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/065_soysauce/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.254,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOYSAUCE_BASE2 = {
    'NAME': 'soysauce_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/065_soysauce/visual/base2.usd",
    'X_LENGTH': 0.115,
    'Y_LENGTH': 0.115,
    'Z_LENGTH': 0.254,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOYSAUCE_BASE3 = {
    'NAME': 'soysauce_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/065_soysauce/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.27,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SOYSAUCE_BASE4 = {
    'NAME': 'soysauce_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/065_soysauce/visual/base4.usd",
    'X_LENGTH': 0.082,
    'Y_LENGTH': 0.083,
    'Z_LENGTH': 0.153,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE0 = {
    'NAME': 'tabletrashbin_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base0.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE1 = {
    'NAME': 'tabletrashbin_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base1.usd",
    'X_LENGTH': 0.14,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE10 = {
    'NAME': 'tabletrashbin_base10',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base10.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.085,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE2 = {
    'NAME': 'tabletrashbin_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base2.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE3 = {
    'NAME': 'tabletrashbin_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base3.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE4 = {
    'NAME': 'tabletrashbin_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base4.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.12,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE5 = {
    'NAME': 'tabletrashbin_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base5.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE6 = {
    'NAME': 'tabletrashbin_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base6.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE7 = {
    'NAME': 'tabletrashbin_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base7.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE8 = {
    'NAME': 'tabletrashbin_base8',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base8.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLETRASHBIN_BASE9 = {
    'NAME': 'tabletrashbin_base9',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/063_tabletrashbin/visual/base9.usd",
    'X_LENGTH': 0.125,
    'Y_LENGTH': 0.125,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE0 = {
    'NAME': 'teabox_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base0.usd",
    'X_LENGTH': 0.151,
    'Y_LENGTH': 0.101,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE1 = {
    'NAME': 'teabox_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base1.usd",
    'X_LENGTH': 0.151,
    'Y_LENGTH': 0.141,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE2 = {
    'NAME': 'teabox_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base2.usd",
    'X_LENGTH': 0.081,
    'Y_LENGTH': 0.061,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE3 = {
    'NAME': 'teabox_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base3.usd",
    'X_LENGTH': 0.151,
    'Y_LENGTH': 0.141,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE4 = {
    'NAME': 'teabox_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base4.usd",
    'X_LENGTH': 0.141,
    'Y_LENGTH': 0.141,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TEABOX_BASE5 = {
    'NAME': 'teabox_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/112_teabox/visual/base5.usd",
    'X_LENGTH': 0.141,
    'Y_LENGTH': 0.141,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOOTHPASTE_BASE0 = {
    'NAME': 'toothpaste_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/118_toothpaste/visual/base0.usd",
    'X_LENGTH': 0.063,
    'Y_LENGTH': 0.031,
    'Z_LENGTH': 0.193,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE0 = {
    'NAME': 'toycar_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base0.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.167,
    'Z_LENGTH': 0.101,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE1 = {
    'NAME': 'toycar_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base1.usd",
    'X_LENGTH': 0.089,
    'Y_LENGTH': 0.171,
    'Z_LENGTH': 0.054,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE2 = {
    'NAME': 'toycar_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base2.usd",
    'X_LENGTH': 0.095,
    'Y_LENGTH': 0.171,
    'Z_LENGTH': 0.093,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE3 = {
    'NAME': 'toycar_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base3.usd",
    'X_LENGTH': 0.075,
    'Y_LENGTH': 0.154,
    'Z_LENGTH': 0.051,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE4 = {
    'NAME': 'toycar_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base4.usd",
    'X_LENGTH': 0.107,
    'Y_LENGTH': 0.172,
    'Z_LENGTH': 0.099,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TOYCAR_BASE5 = {
    'NAME': 'toycar_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/057_toycar/visual/base5.usd",
    'X_LENGTH': 0.102,
    'Y_LENGTH': 0.171,
    'Z_LENGTH': 0.085,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

VINEGAR_BASE0 = {
    'NAME': 'vinegar_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/066_vinegar/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.24,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

VINEGAR_BASE1 = {
    'NAME': 'vinegar_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/066_vinegar/visual/base1.usd",
    'X_LENGTH': 0.088,
    'Y_LENGTH': 0.088,
    'Z_LENGTH': 0.255,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

VINEGAR_BASE2 = {
    'NAME': 'vinegar_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/066_vinegar/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.255,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE0 = {
    'NAME': 'waterer_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base0.usd",
    'X_LENGTH': 0.101,
    'Y_LENGTH': 0.207,
    'Z_LENGTH': 0.101,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE1 = {
    'NAME': 'waterer_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base1.usd",
    'X_LENGTH': 0.101,
    'Y_LENGTH': 0.204,
    'Z_LENGTH': 0.101,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE2 = {
    'NAME': 'waterer_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base2.usd",
    'X_LENGTH': 0.072,
    'Y_LENGTH': 0.201,
    'Z_LENGTH': 0.142,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE3 = {
    'NAME': 'waterer_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base3.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.218,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE4 = {
    'NAME': 'waterer_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base4.usd",
    'X_LENGTH': 0.081,
    'Y_LENGTH': 0.218,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE5 = {
    'NAME': 'waterer_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base5.usd",
    'X_LENGTH': 0.086,
    'Y_LENGTH': 0.195,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE6 = {
    'NAME': 'waterer_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base6.usd",
    'X_LENGTH': 0.079,
    'Y_LENGTH': 0.185,
    'Z_LENGTH': 0.125,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WATERER_BASE7 = {
    'NAME': 'waterer_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/087_waterer/visual/base7.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.185,
    'Z_LENGTH': 0.135,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WHITEBOARD_ERASER_BASE0 = {
    'NAME': 'whiteboard_eraser_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/117_whiteboard_eraser/visual/base0.usd",
    'X_LENGTH': 0.114,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.02,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WINEGLASS_BASE0 = {
    'NAME': 'wineglass_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/088_wineglass/visual/base0.usd",
    'X_LENGTH': 0.115,
    'Y_LENGTH': 0.115,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WINEGLASS_BASE1 = {
    'NAME': 'wineglass_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/088_wineglass/visual/base1.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WINEGLASS_BASE2 = {
    'NAME': 'wineglass_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/088_wineglass/visual/base2.usd",
    'X_LENGTH': 0.072,
    'Y_LENGTH': 0.072,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WINEGLASS_BASE3 = {
    'NAME': 'wineglass_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/088_wineglass/visual/base3.usd",
    'X_LENGTH': 0.072,
    'Y_LENGTH': 0.072,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WINEGLASS_BASE4 = {
    'NAME': 'wineglass_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/088_wineglass/visual/base4.usd",
    'X_LENGTH': 0.086,
    'Y_LENGTH': 0.086,
    'Z_LENGTH': 0.195,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODENBLOCK_BASE0 = {
    'NAME': 'woodenblock_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/086_woodenblock/visual/base0.usd",
    'X_LENGTH': 0.102,
    'Y_LENGTH': 0.102,
    'Z_LENGTH': 0.102,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODENBLOCK_BASE1 = {
    'NAME': 'woodenblock_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/086_woodenblock/visual/base1.usd",
    'X_LENGTH': 0.071,
    'Y_LENGTH': 0.071,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODENBLOCK_BASE2 = {
    'NAME': 'woodenblock_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/086_woodenblock/visual/base2.usd",
    'X_LENGTH': 0.131,
    'Y_LENGTH': 0.061,
    'Z_LENGTH': 0.131,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODENBLOCK_BASE3 = {
    'NAME': 'woodenblock_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/086_woodenblock/visual/base3.usd",
    'X_LENGTH': 0.151,
    'Y_LENGTH': 0.037,
    'Z_LENGTH': 0.052,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODENBLOCK_BASE4 = {
    'NAME': 'woodenblock_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_adapted/086_woodenblock/visual/base4.usd",
    'X_LENGTH': 0.121,
    'Y_LENGTH': 0.122,
    'Z_LENGTH': 0.153,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

# 统一的 SKU 配置表（每个变体独立挂在这里）
SKU_CONFIG = {
    CRACKER_BOX_PARAMS['NAME']: CRACKER_BOX_PARAMS,
    SUGER_BOX_PARAMS['NAME']: SUGER_BOX_PARAMS,
    PLASTIC_PACKAGE_PARAMS['NAME']: PLASTIC_PACKAGE_PARAMS,
    SF_BIG_PARAMS['NAME']: SF_BIG_PARAMS,
    SF_SMALL_PARAMS['NAME']: SF_SMALL_PARAMS,
    BOTTLE_BASE1['NAME']: BOTTLE_BASE1,
    BOTTLE_BASE2['NAME']: BOTTLE_BASE2,
    BOTTLE_BASE3['NAME']: BOTTLE_BASE3,
    BOTTLE_BASE4['NAME']: BOTTLE_BASE4,
    BOXDRINK_BASE0['NAME']: BOXDRINK_BASE0,
    BOXDRINK_BASE1['NAME']: BOXDRINK_BASE1,
    BOXDRINK_BASE2['NAME']: BOXDRINK_BASE2,
    BOXDRINK_BASE3['NAME']: BOXDRINK_BASE3,
    CAN_BASE0['NAME']: CAN_BASE0,
    CAN_BASE1['NAME']: CAN_BASE1,
    CAN_BASE2['NAME']: CAN_BASE2,
    CAN_BASE3['NAME']: CAN_BASE3,
    CAN_BASE5['NAME']: CAN_BASE5,
    CAN_BASE6['NAME']: CAN_BASE6,
    COFFEEBOX_BASE0['NAME']: COFFEEBOX_BASE0,
    COFFEEBOX_BASE1['NAME']: COFFEEBOX_BASE1,
    COFFEEBOX_BASE2['NAME']: COFFEEBOX_BASE2,
    COFFEEBOX_BASE3['NAME']: COFFEEBOX_BASE3,
    COFFEEBOX_BASE4['NAME']: COFFEEBOX_BASE4,
    COFFEEBOX_BASE5['NAME']: COFFEEBOX_BASE5,
    COFFEEBOX_BASE6['NAME']: COFFEEBOX_BASE6,
    FAN_BASE0['NAME']: FAN_BASE0,
    FAN_BASE1['NAME']: FAN_BASE1,
    FAN_BASE2['NAME']: FAN_BASE2,
    FAN_BASE3['NAME']: FAN_BASE3,
    FAN_BASE4['NAME']: FAN_BASE4,
    FAN_BASE5['NAME']: FAN_BASE5,
    FAN_BASE6['NAME']: FAN_BASE6,
    GLUE_BASE0['NAME']: GLUE_BASE0,
    GLUE_BASE1['NAME']: GLUE_BASE1,
    GLUE_BASE2['NAME']: GLUE_BASE2,
    GLUE_BASE4['NAME']: GLUE_BASE4,
    GLUE_BASE5['NAME']: GLUE_BASE5,
    GLUE_BASE6['NAME']: GLUE_BASE6,
    KETTLE_BASE0['NAME']: KETTLE_BASE0,
    KETTLE_BASE1['NAME']: KETTLE_BASE1,
    KETTLE_BASE2['NAME']: KETTLE_BASE2,
    KETTLE_BASE3['NAME']: KETTLE_BASE3,
    KETTLE_BASE4['NAME']: KETTLE_BASE4,
    KETTLE_BASE5['NAME']: KETTLE_BASE5,
    KEYBOARD_BASE0['NAME']: KEYBOARD_BASE0,
    KEYBOARD_BASE1['NAME']: KEYBOARD_BASE1,
    KEYBOARD_BASE2['NAME']: KEYBOARD_BASE2,
    KEYBOARD_BASE3['NAME']: KEYBOARD_BASE3,
    MILKTEA_BASE0['NAME']: MILKTEA_BASE0,
    MILKTEA_BASE1['NAME']: MILKTEA_BASE1,
    MILKTEA_BASE2['NAME']: MILKTEA_BASE2,
    MILKTEA_BASE4['NAME']: MILKTEA_BASE4,
    MILKTEA_BASE5['NAME']: MILKTEA_BASE5,
    MILKTEA_BASE6['NAME']: MILKTEA_BASE6,
    MSG_BASE0['NAME']: MSG_BASE0,
    MSG_BASE1['NAME']: MSG_BASE1,
    MSG_BASE2['NAME']: MSG_BASE2,
    MSG_BASE3['NAME']: MSG_BASE3,
    MSG_BASE4['NAME']: MSG_BASE4,
    MSG_BASE5['NAME']: MSG_BASE5,
    NOTEBOOK_BASE0['NAME']: NOTEBOOK_BASE0,
    NOTEBOOK_BASE1['NAME']: NOTEBOOK_BASE1,
    NOTEBOOK_BASE2['NAME']: NOTEBOOK_BASE2,
    PERFUME_BASE0['NAME']: PERFUME_BASE0,
    PERFUME_BASE1['NAME']: PERFUME_BASE1,
    PERFUME_BASE2['NAME']: PERFUME_BASE2,
    PERFUME_BASE3['NAME']: PERFUME_BASE3,
    PHONE_BASE0['NAME']: PHONE_BASE0,
    PHONE_BASE1['NAME']: PHONE_BASE1,
    PHONE_BASE2['NAME']: PHONE_BASE2,
    PHONE_BASE3['NAME']: PHONE_BASE3,
    PHONE_BASE4['NAME']: PHONE_BASE4,
    PILLBOTTLE_BASE1['NAME']: PILLBOTTLE_BASE1,
    PILLBOTTLE_BASE2['NAME']: PILLBOTTLE_BASE2,
    PILLBOTTLE_BASE3['NAME']: PILLBOTTLE_BASE3,
    PILLBOTTLE_BASE4['NAME']: PILLBOTTLE_BASE4,
    PILLBOTTLE_BASE5['NAME']: PILLBOTTLE_BASE5,
    PLASTICBOX_BASE0['NAME']: PLASTICBOX_BASE0,
    PLASTICBOX_BASE1['NAME']: PLASTICBOX_BASE1,
    PLASTICBOX_BASE10['NAME']: PLASTICBOX_BASE10,
    PLASTICBOX_BASE2['NAME']: PLASTICBOX_BASE2,
    PLASTICBOX_BASE3['NAME']: PLASTICBOX_BASE3,
    PLASTICBOX_BASE4['NAME']: PLASTICBOX_BASE4,
    PLASTICBOX_BASE5['NAME']: PLASTICBOX_BASE5,
    PLASTICBOX_BASE6['NAME']: PLASTICBOX_BASE6,
    PLASTICBOX_BASE7['NAME']: PLASTICBOX_BASE7,
    PLASTICBOX_BASE8['NAME']: PLASTICBOX_BASE8,
    PLASTICBOX_BASE9['NAME']: PLASTICBOX_BASE9,
    PLAYINGCARDS_BASE0['NAME']: PLAYINGCARDS_BASE0,
    PLAYINGCARDS_BASE1['NAME']: PLAYINGCARDS_BASE1,
    PLAYINGCARDS_BASE2['NAME']: PLAYINGCARDS_BASE2,
    REMOTECONTROL_BASE0['NAME']: REMOTECONTROL_BASE0,
    REMOTECONTROL_BASE1['NAME']: REMOTECONTROL_BASE1,
    REMOTECONTROL_BASE2['NAME']: REMOTECONTROL_BASE2,
    REMOTECONTROL_BASE3['NAME']: REMOTECONTROL_BASE3,
    REMOTECONTROL_BASE4['NAME']: REMOTECONTROL_BASE4,
    REMOTECONTROL_BASE5['NAME']: REMOTECONTROL_BASE5,
    REMOTECONTROL_BASE6['NAME']: REMOTECONTROL_BASE6,
    REST_BASE0['NAME']: REST_BASE0,
    REST_BASE1['NAME']: REST_BASE1,
    REST_BASE2['NAME']: REST_BASE2,
    REST_BASE3['NAME']: REST_BASE3,
    ROLLER_BASE0['NAME']: ROLLER_BASE0,
    ROLLER_BASE1['NAME']: ROLLER_BASE1,
    ROLLER_BASE2['NAME']: ROLLER_BASE2,
    RUBIKSCUBE_BASE0['NAME']: RUBIKSCUBE_BASE0,
    RUBIKSCUBE_BASE1['NAME']: RUBIKSCUBE_BASE1,
    RUBIKSCUBE_BASE2['NAME']: RUBIKSCUBE_BASE2,
    SAUCECAN_BASE0['NAME']: SAUCECAN_BASE0,
    SAUCECAN_BASE2['NAME']: SAUCECAN_BASE2,
    SAUCECAN_BASE4['NAME']: SAUCECAN_BASE4,
    SAUCECAN_BASE5['NAME']: SAUCECAN_BASE5,
    SAUCECAN_BASE6['NAME']: SAUCECAN_BASE6,
    SMALLSPEAKER_BASE1['NAME']: SMALLSPEAKER_BASE1,
    SMALLSPEAKER_BASE2['NAME']: SMALLSPEAKER_BASE2,
    SMALLSPEAKER_BASE3['NAME']: SMALLSPEAKER_BASE3,
    SOAP_BASE0['NAME']: SOAP_BASE0,
    SOAP_BASE1['NAME']: SOAP_BASE1,
    SOAP_BASE2['NAME']: SOAP_BASE2,
    SOAP_BASE3['NAME']: SOAP_BASE3,
    SOYSAUCE_BASE0['NAME']: SOYSAUCE_BASE0,
    SOYSAUCE_BASE1['NAME']: SOYSAUCE_BASE1,
    SOYSAUCE_BASE2['NAME']: SOYSAUCE_BASE2,
    SOYSAUCE_BASE3['NAME']: SOYSAUCE_BASE3,
    SOYSAUCE_BASE4['NAME']: SOYSAUCE_BASE4,
    TABLETRASHBIN_BASE0['NAME']: TABLETRASHBIN_BASE0,
    TABLETRASHBIN_BASE1['NAME']: TABLETRASHBIN_BASE1,
    TABLETRASHBIN_BASE10['NAME']: TABLETRASHBIN_BASE10,
    TABLETRASHBIN_BASE2['NAME']: TABLETRASHBIN_BASE2,
    TABLETRASHBIN_BASE3['NAME']: TABLETRASHBIN_BASE3,
    TABLETRASHBIN_BASE4['NAME']: TABLETRASHBIN_BASE4,
    TABLETRASHBIN_BASE5['NAME']: TABLETRASHBIN_BASE5,
    TABLETRASHBIN_BASE6['NAME']: TABLETRASHBIN_BASE6,
    TABLETRASHBIN_BASE7['NAME']: TABLETRASHBIN_BASE7,
    TABLETRASHBIN_BASE8['NAME']: TABLETRASHBIN_BASE8,
    TABLETRASHBIN_BASE9['NAME']: TABLETRASHBIN_BASE9,
    TEABOX_BASE0['NAME']: TEABOX_BASE0,
    TEABOX_BASE1['NAME']: TEABOX_BASE1,
    TEABOX_BASE2['NAME']: TEABOX_BASE2,
    TEABOX_BASE3['NAME']: TEABOX_BASE3,
    TEABOX_BASE4['NAME']: TEABOX_BASE4,
    TEABOX_BASE5['NAME']: TEABOX_BASE5,
    TOOTHPASTE_BASE0['NAME']: TOOTHPASTE_BASE0,
    TOYCAR_BASE0['NAME']: TOYCAR_BASE0,
    TOYCAR_BASE1['NAME']: TOYCAR_BASE1,
    TOYCAR_BASE2['NAME']: TOYCAR_BASE2,
    TOYCAR_BASE3['NAME']: TOYCAR_BASE3,
    TOYCAR_BASE4['NAME']: TOYCAR_BASE4,
    TOYCAR_BASE5['NAME']: TOYCAR_BASE5,
    VINEGAR_BASE0['NAME']: VINEGAR_BASE0,
    VINEGAR_BASE1['NAME']: VINEGAR_BASE1,
    VINEGAR_BASE2['NAME']: VINEGAR_BASE2,
    WATERER_BASE0['NAME']: WATERER_BASE0,
    WATERER_BASE1['NAME']: WATERER_BASE1,
    WATERER_BASE2['NAME']: WATERER_BASE2,
    WATERER_BASE3['NAME']: WATERER_BASE3,
    WATERER_BASE4['NAME']: WATERER_BASE4,
    WATERER_BASE5['NAME']: WATERER_BASE5,
    WATERER_BASE6['NAME']: WATERER_BASE6,
    WATERER_BASE7['NAME']: WATERER_BASE7,
    WHITEBOARD_ERASER_BASE0['NAME']: WHITEBOARD_ERASER_BASE0,
    WINEGLASS_BASE0['NAME']: WINEGLASS_BASE0,
    WINEGLASS_BASE1['NAME']: WINEGLASS_BASE1,
    WINEGLASS_BASE2['NAME']: WINEGLASS_BASE2,
    WINEGLASS_BASE3['NAME']: WINEGLASS_BASE3,
    WINEGLASS_BASE4['NAME']: WINEGLASS_BASE4,
    WOODENBLOCK_BASE0['NAME']: WOODENBLOCK_BASE0,
    WOODENBLOCK_BASE1['NAME']: WOODENBLOCK_BASE1,
    WOODENBLOCK_BASE2['NAME']: WOODENBLOCK_BASE2,
    WOODENBLOCK_BASE3['NAME']: WOODENBLOCK_BASE3,
    WOODENBLOCK_BASE4['NAME']: WOODENBLOCK_BASE4
}
