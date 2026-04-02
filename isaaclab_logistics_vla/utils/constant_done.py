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

ALARM_CLOCK_BASE0 = {
    'NAME': 'alarm_clock_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base0.usd",
    'X_LENGTH': 0.19,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ALARM_CLOCK_BASE1 = {
    'NAME': 'alarm_clock_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base1.usd",
    'X_LENGTH': 0.19,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ALARM_CLOCK_BASE2 = {
    'NAME': 'alarm_clock_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base2.usd",
    'X_LENGTH': 0.17,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ALARM_CLOCK_BASE3 = {
    'NAME': 'alarm_clock_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base3.usd",
    'X_LENGTH': 0.17,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ALARM_CLOCK_BASE4 = {
    'NAME': 'alarm_clock_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base4.usd",
    'X_LENGTH': 0.18,
    'Y_LENGTH': 0.06,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ALARM_CLOCK_BASE5 = { 
    'NAME': 'alarm_clock_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/046_alarm_clock/visual/base5.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

APPLE_BASE0 = {
    'NAME': 'apple_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/035_apple/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

APPLE_BASE1 = { #不要苹果
    'NAME': 'apple_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/035_apple/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOOK_BASE0 = {
    'NAME': 'book_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/043_book/visual/base0.usd",
    'X_LENGTH': 0.16,
    'Y_LENGTH': 0.21,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 90),(0, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

BOOK_BASE1 = {
    'NAME': 'book_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/043_book/visual/base1.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 90),(0, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE1 = {
    'NAME': 'bowl_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE2 = {
    'NAME': 'bowl_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.1,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE3 = {
    'NAME': 'bowl_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE4 = {
    'NAME': 'bowl_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base4.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE5 = {
    'NAME': 'bowl_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base5.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE6 = { #有问题，感觉生成的位置不对
    'NAME': 'bowl_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base6.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (180, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

BOWL_BASE7 = { #有问题，感觉生成的位置不对
    'NAME': 'bowl_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/002_bowl/visual/base7.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.03,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (180, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE0 = {
    'NAME': 'calculator_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.14,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE1 = { # 原始文件中坐标轴不正导致生成姿态不平行于地面
    'NAME': 'calculator_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base1.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE2 = {
    'NAME': 'calculator_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base2.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.14,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE3 = {
    'NAME': 'calculator_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.12,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE4 = {
    'NAME': 'calculator_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base4.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.04,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CALCULATOR_BASE5 = {
    'NAME': 'calculator_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/017_calculator/visual/base5.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 90), (180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CHIPS_TUB_BASE0 = { 
    'NAME': 'chips_tub_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/025_chips_tub/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.15,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CHIPS_TUB_BASE1 = {
    'NAME': 'chips_tub_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/025_chips_tub/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.23,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CHIPS_TUB_BASE2 = {
    'NAME': 'chips_tub_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/025_chips_tub/visual/base2.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.18,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CHIPS_TUB_BASE3 = {
    'NAME': 'chips_tub_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/025_chips_tub/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.23,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90,0,0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE0 = {
    'NAME': 'cup_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE1 = {
    'NAME': 'cup_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE2 = {
    'NAME': 'cup_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base2.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE3 = {
    'NAME': 'cup_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE4 = {
    'NAME': 'cup_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base4.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE5 = {
    'NAME': 'cup_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base5.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

CUP_BASE6 = {
    'NAME': 'cup_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base6.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE7 = {
    'NAME': 'cup_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base7.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT':[(0, 0, 0),(0,90,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE8 = {
    'NAME': 'cup_base8',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base8.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.1,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE9 = {
    'NAME': 'cup_base9',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base9.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE10 = {
    'NAME': 'cup_base10',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base10.usd",
    'X_LENGTH': 0.08,
    'Z_LENGTH': 0.06,
    'Y_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE11 = {
    'NAME': 'cup_base11',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base11.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

CUP_BASE12 = {
    'NAME': 'cup_base12',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/021_cup/visual/base12.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

DUSTBIN_BASE0 = {
    'NAME': 'dustbin_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/011_dustbin/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 90),(0,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FLUTED_BLOCK_BASE0 = {
    'NAME': 'fluted_block_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/004_fluted_block/visual/base0.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

FLUTED_BLOCK_BASE1 = {
    'NAME': 'fluted_block_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/004_fluted_block/visual/base1.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (180, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

JAM_JAR_BASE0 = {
    'NAME': 'jam_jar_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/031_jam_jar/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(180,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

JAM_JAR_BASE1 = {
    'NAME': 'jam_jar_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/031_jam_jar/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(180,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

JAM_JAR_BASE2 = {
    'NAME': 'jam_jar_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/031_jam_jar/visual/base2.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(180,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

JAM_JAR_BASE3 = {
    'NAME': 'jam_jar_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/031_jam_jar/visual/base3.usd",
    'Y_LENGTH': 0.08,
    'X_LENGTH': 0.13,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(-90, 0, 0),(180, 0, 0),(180, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

JAM_JAR_BASE4 = {
    'NAME': 'jam_jar_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/031_jam_jar/visual/base4.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(180,0,0)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

KETTLE_BASE0 = { #水壶，形态有问题
    'NAME': 'kettle_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/009_kettle/visual/base0.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE1 = {
    'NAME': 'kettle_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/009_kettle/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.14,
    'Z_LENGTH': 0.12,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

KETTLE_BASE2 = {
    'NAME': 'kettle_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/009_kettle/visual/base2.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MICROPHONE_BASE0 = {
    'NAME': 'microphone_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/018_microphone/visual/base0.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90, 90, 0),(90, 90, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MICROPHONE_BASE1 = {
    'NAME': 'microphone_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/018_microphone/visual/base1.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0),(0, 90, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MICROPHONE_BASE4 = {
    'NAME': 'microphone_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/018_microphone/visual/base4.usd",
    'X_LENGTH': 0.19,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MICROPHONE_BASE5 = {
    'NAME': 'microphone_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/018_microphone/visual/base5.usd",
    'X_LENGTH': 0.04,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.04,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(0,90,0),(0, 90, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MILK_BOX_BASE0 = {
    'NAME': 'milk_box_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/038_milk_box/visual/base0.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.14,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,90,0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MILK_BOX_BASE1 = {
    'NAME': 'milk_box_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/038_milk_box/visual/base1.usd",
    'Y_LENGTH': 0.06,
    'X_LENGTH': 0.09,
    'Z_LENGTH': 0.2,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MILK_BOX_BASE2 = {
    'NAME': 'milk_box_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/038_milk_box/visual/base2.usd",
    'X_LENGTH': 0.06,
    'Z_LENGTH': 0.17,
    'Y_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,90,0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MILK_BOX_BASE3 = {
    'NAME': 'milk_box_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/038_milk_box/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Z_LENGTH': 0.14,
    'Y_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,90,0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

MOUSE_BASE0 = {
    'NAME': 'mouse_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/047_mouse/visual/base0.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.17,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MOUSE_BASE1 = {
    'NAME': 'mouse_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/047_mouse/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MOUSE_BASE2 = {
    'NAME': 'mouse_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/047_mouse/visual/base2.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.12,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180, 0, 90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

MUG_BASE0 = {
    'NAME': 'mug_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base0.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE1 = {
    'NAME': 'mug_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE10 = {
    'NAME': 'mug_base10',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base10.usd",
    'X_LENGTH': 0.09,
    'Z_LENGTH': 0.1,
    'Y_LENGTH': 0.13,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE11 = {
    'NAME': 'mug_base11',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base11.usd",
    'X_LENGTH': 0.09,
    'Z_LENGTH': 0.07,
    'Y_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE12 = {
    'NAME': 'mug_base12',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base12.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE2 = {
    'NAME': 'mug_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base2.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE3 = {
    'NAME': 'mug_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE4 = {
    'NAME': 'mug_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base4.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE5 = {
    'NAME': 'mug_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base5.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE6 = {
    'NAME': 'mug_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base6.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE7 = {
    'NAME': 'mug_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base7.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE8 = {
    'NAME': 'mug_base8',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base8.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

MUG_BASE9 = {
    'NAME': 'mug_base9',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/039_mug/visual/base9.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.11,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

OLIVE_OIL_BASE0 = {
    'NAME': 'olive_oil_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/029_olive_oil/visual/base0.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.21,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (180, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

OLIVE_OIL_BASE1 = {
    'NAME': 'olive_oil_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/029_olive_oil/visual/base1.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.3,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (180, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

OLIVE_OIL_BASE2 = {
    'NAME': 'olive_oil_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/029_olive_oil/visual/base2.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.3,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

OLIVE_OIL_BASE3 = {
    'NAME': 'olive_oil_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/029_olive_oil/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.07,
    'Z_LENGTH': 0.3,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

OLIVE_OIL_BASE4 = {
    'NAME': 'olive_oil_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/029_olive_oil/visual/base4.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.3,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0), (180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE0 = {
    'NAME': 'plastic_bottle_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base0.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE1 = {
    'NAME': 'plastic_bottle_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base1.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE10 = {
    'NAME': 'plastic_bottle_base10',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base10.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE11 = {
    'NAME': 'plastic_bottle_base11',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base11.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.24,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE12 = {
    'NAME': 'plastic_bottle_base12',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base12.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE13 = {
    'NAME': 'plastic_bottle_base13',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base13.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.27,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE14 = {
    'NAME': 'plastic_bottle_base14',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base14.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE15 = {
    'NAME': 'plastic_bottle_base15',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base15.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE16 = {
    'NAME': 'plastic_bottle_base16',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base16.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE17 = {
    'NAME': 'plastic_bottle_base17',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base17.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE18 = {
    'NAME': 'plastic_bottle_base18',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base18.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.21,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE19 = {
    'NAME': 'plastic_bottle_base19',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base19.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE2 = {
    'NAME': 'plastic_bottle_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base2.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.24,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE20 = {
    'NAME': 'plastic_bottle_base20',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base20.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE21 = {
    'NAME': 'plastic_bottle_base21',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base21.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE22 = {
    'NAME': 'plastic_bottle_base22',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base22.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.26,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE3 = {
    'NAME': 'plastic_bottle_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base3.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE4 = {
    'NAME': 'plastic_bottle_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base4.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.26,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE5 = {
    'NAME': 'plastic_bottle_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base5.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.8
}

PLASTIC_BOTTLE_BASE6 = {
    'NAME': 'plastic_bottle_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base6.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE7 = {
    'NAME': 'plastic_bottle_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base7.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.28,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE8 = {
    'NAME': 'plastic_bottle_base8',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base8.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.21,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLASTIC_BOTTLE_BASE9 = {
    'NAME': 'plastic_bottle_base9',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/001_plastic_bottle/visual/base9.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.23,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(0, 0, 0), (0,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.7
}

PLATE_BASE0 = {
    'NAME': 'plate_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/003_plate/visual/base0.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.01,
    'Z_LENGTH': 0.1,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ROLL_PAPER_BASE0 = {
    'NAME': 'roll_paper_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/028_roll_paper/visual/base0.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

ROLL_PAPER_BASE1 = {
    'NAME': 'roll_paper_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/028_roll_paper/visual/base1.usd",
    'X_LENGTH': 0.1,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

ROLL_PAPER_BASE2 = {
    'NAME': 'roll_paper_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/028_roll_paper/visual/base2.usd",
    'X_LENGTH': 0.11,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

ROLL_PAPER_BASE3 = {
    'NAME': 'roll_paper_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/028_roll_paper/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

SHAMPOO_BASE1 = {
    'NAME': 'shampoo_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base1.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.04,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE2 = {
    'NAME': 'shampoo_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base2.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.04,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE3 = {
    'NAME': 'shampoo_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base3.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE4 = {
    'NAME': 'shampoo_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base4.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE5 = {
    'NAME': 'shampoo_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base5.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE6 = {
    'NAME': 'shampoo_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base6.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHAMPOO_BASE7 = {
    'NAME': 'shampoo_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/049_shampoo/visual/base7.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.19,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (90, 0, 90),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

SHOE_BOX_BASE0 = { # 放进去环境变暗
    'NAME': 'shoe_box_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/007_shoe_box/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.16,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE1 = {
    'NAME': 'stapler_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base1.usd",
    'X_LENGTH': 0.04,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE2 = {
    'NAME': 'stapler_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base2.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE3 = {
    'NAME': 'stapler_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base3.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.18,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE4 = {
    'NAME': 'stapler_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base4.usd",
    'X_LENGTH': 0.05,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE5 = {
    'NAME': 'stapler_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base5.usd",
    'X_LENGTH': 0.06,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE6 = {
    'NAME': 'stapler_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base6.usd",
    'X_LENGTH': 0.04,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

STAPLER_BASE7 = {
    'NAME': 'stapler_base7',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/048_stapler/visual/base7.usd",
    'X_LENGTH': 0.07,
    'Y_LENGTH': 0.19,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLE_TENNIS_BASE0 = {
    'NAME': 'table_tennis_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/027_table_tennis/visual/base0.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TABLE_TENNIS_BASE1 = {
    'NAME': 'table_tennis_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/027_table_tennis/visual/base1.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.09,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': (0, 0, 0),
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TISSUE_BOX_BASE0 = {
    'NAME': 'tissue_box_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base0.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE1 = {
    'NAME': 'tissue_box_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base1.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE2 = {
    'NAME': 'tissue_box_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base2.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE3 = {
    'NAME': 'tissue_box_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base3.usd",
    'X_LENGTH': 0.09,
    'Y_LENGTH': 0.15,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE4 = {
    'NAME': 'tissue_box_base4',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base4.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.08,
    'Z_LENGTH': 0.05,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE5 = {
    'NAME': 'tissue_box_base5',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base5.usd",
    'X_LENGTH': 0.14,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.06,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TISSUE_BOX_BASE6 = {
    'NAME': 'tissue_box_base6',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/023_tissue_box/visual/base6.usd",
    'X_LENGTH': 0.08,
    'Y_LENGTH': 0.11,
    'Z_LENGTH': 0.07,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 0.9
}

TRAY_BASE0 = {
    'NAME': 'tray_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/008_tray/visual/base0.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TRAY_BASE1 = {
    'NAME': 'tray_base1',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/008_tray/visual/base1.usd",
    'X_LENGTH': 0.15,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TRAY_BASE2 = {
    'NAME': 'tray_base2',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/008_tray/visual/base2.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.09,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

TRAY_BASE3 = {
    'NAME': 'tray_base3',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/008_tray/visual/base3.usd",
    'X_LENGTH': 0.13,
    'Y_LENGTH': 0.1,
    'Z_LENGTH': 0.01,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(180, 0, 0),(180,0,90)],
    'STACK_SCALE': 1,
    'SPARSE_SCALE': 1,
    'DENSE_SCALE': 1
}

WOODEN_BOX_BASE0 = {
    'NAME': 'wooden_box_base0',
    'USD_PATH': f"{ASSET_ROOT_PATH}/objects_center/042_wooden_box/visual/base0.usd",
    'X_LENGTH': 0.12,
    'Y_LENGTH': 0.05,
    'Z_LENGTH': 0.08,
    'SPARSE_ORIENT': (0, 0, 0),
    'STACK_ORIENT': (0, 0, 0),
    'DENSE_ORIENT': [(90, 0, 0),(90,0,90)],
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
    ALARM_CLOCK_BASE0['NAME']: ALARM_CLOCK_BASE0,
    ALARM_CLOCK_BASE1['NAME']: ALARM_CLOCK_BASE1,
    ALARM_CLOCK_BASE2['NAME']: ALARM_CLOCK_BASE2,
    ALARM_CLOCK_BASE3['NAME']: ALARM_CLOCK_BASE3,
    ALARM_CLOCK_BASE4['NAME']: ALARM_CLOCK_BASE4,
    ALARM_CLOCK_BASE5['NAME']: ALARM_CLOCK_BASE5,
    APPLE_BASE0['NAME']: APPLE_BASE0,
    APPLE_BASE1['NAME']: APPLE_BASE1,
    BOOK_BASE0['NAME']: BOOK_BASE0,
    BOOK_BASE1['NAME']: BOOK_BASE1,
    BOWL_BASE1['NAME']: BOWL_BASE1,
    BOWL_BASE2['NAME']: BOWL_BASE2,
    BOWL_BASE3['NAME']: BOWL_BASE3,
    BOWL_BASE4['NAME']: BOWL_BASE4,
    BOWL_BASE5['NAME']: BOWL_BASE5,
    BOWL_BASE6['NAME']: BOWL_BASE6,
    BOWL_BASE7['NAME']: BOWL_BASE7,
    CALCULATOR_BASE0['NAME']: CALCULATOR_BASE0,
    CALCULATOR_BASE1['NAME']: CALCULATOR_BASE1,
    CALCULATOR_BASE2['NAME']: CALCULATOR_BASE2,
    CALCULATOR_BASE3['NAME']: CALCULATOR_BASE3,
    CALCULATOR_BASE4['NAME']: CALCULATOR_BASE4,
    CALCULATOR_BASE5['NAME']: CALCULATOR_BASE5,
    CHIPS_TUB_BASE0['NAME']: CHIPS_TUB_BASE0,
    CHIPS_TUB_BASE1['NAME']: CHIPS_TUB_BASE1,
    CHIPS_TUB_BASE2['NAME']: CHIPS_TUB_BASE2,
    CHIPS_TUB_BASE3['NAME']: CHIPS_TUB_BASE3,
    CUP_BASE0['NAME']: CUP_BASE0,
    CUP_BASE1['NAME']: CUP_BASE1,
    CUP_BASE10['NAME']: CUP_BASE10,
    CUP_BASE11['NAME']: CUP_BASE11,
    CUP_BASE12['NAME']: CUP_BASE12,
    CUP_BASE2['NAME']: CUP_BASE2,
    CUP_BASE3['NAME']: CUP_BASE3,
    CUP_BASE4['NAME']: CUP_BASE4,
    CUP_BASE5['NAME']: CUP_BASE5,
    CUP_BASE6['NAME']: CUP_BASE6,
    CUP_BASE7['NAME']: CUP_BASE7,
    CUP_BASE8['NAME']: CUP_BASE8,
    CUP_BASE9['NAME']: CUP_BASE9,
    DUSTBIN_BASE0['NAME']: DUSTBIN_BASE0,
    FLUTED_BLOCK_BASE0['NAME']: FLUTED_BLOCK_BASE0,
    FLUTED_BLOCK_BASE1['NAME']: FLUTED_BLOCK_BASE1,
    JAM_JAR_BASE0['NAME']: JAM_JAR_BASE0,
    JAM_JAR_BASE1['NAME']: JAM_JAR_BASE1,
    JAM_JAR_BASE2['NAME']: JAM_JAR_BASE2,
    JAM_JAR_BASE3['NAME']: JAM_JAR_BASE3,
    JAM_JAR_BASE4['NAME']: JAM_JAR_BASE4,
    KETTLE_BASE0['NAME']: KETTLE_BASE0,
    KETTLE_BASE1['NAME']: KETTLE_BASE1,
    KETTLE_BASE2['NAME']: KETTLE_BASE2,
    MICROPHONE_BASE0['NAME']: MICROPHONE_BASE0,
    MICROPHONE_BASE1['NAME']: MICROPHONE_BASE1,
    MICROPHONE_BASE4['NAME']: MICROPHONE_BASE4,
    MICROPHONE_BASE5['NAME']: MICROPHONE_BASE5,
    MILK_BOX_BASE0['NAME']: MILK_BOX_BASE0,
    MILK_BOX_BASE1['NAME']: MILK_BOX_BASE1,
    MILK_BOX_BASE2['NAME']: MILK_BOX_BASE2,
    MILK_BOX_BASE3['NAME']: MILK_BOX_BASE3,
    MOUSE_BASE0['NAME']: MOUSE_BASE0,
    MOUSE_BASE1['NAME']: MOUSE_BASE1,
    MOUSE_BASE2['NAME']: MOUSE_BASE2,
    MUG_BASE0['NAME']: MUG_BASE0,
    MUG_BASE1['NAME']: MUG_BASE1,
    MUG_BASE10['NAME']: MUG_BASE10,
    MUG_BASE11['NAME']: MUG_BASE11,
    MUG_BASE12['NAME']: MUG_BASE12,
    MUG_BASE2['NAME']: MUG_BASE2,
    MUG_BASE3['NAME']: MUG_BASE3,
    MUG_BASE4['NAME']: MUG_BASE4,
    MUG_BASE5['NAME']: MUG_BASE5,
    MUG_BASE6['NAME']: MUG_BASE6,
    MUG_BASE7['NAME']: MUG_BASE7,
    MUG_BASE8['NAME']: MUG_BASE8,
    MUG_BASE9['NAME']: MUG_BASE9,
    OLIVE_OIL_BASE0['NAME']: OLIVE_OIL_BASE0,
    OLIVE_OIL_BASE1['NAME']: OLIVE_OIL_BASE1,
    OLIVE_OIL_BASE2['NAME']: OLIVE_OIL_BASE2,
    OLIVE_OIL_BASE3['NAME']: OLIVE_OIL_BASE3,
    OLIVE_OIL_BASE4['NAME']: OLIVE_OIL_BASE4,
    PLASTIC_BOTTLE_BASE0['NAME']: PLASTIC_BOTTLE_BASE0,
    PLASTIC_BOTTLE_BASE1['NAME']: PLASTIC_BOTTLE_BASE1,
    PLASTIC_BOTTLE_BASE10['NAME']: PLASTIC_BOTTLE_BASE10,
    PLASTIC_BOTTLE_BASE11['NAME']: PLASTIC_BOTTLE_BASE11,
    PLASTIC_BOTTLE_BASE12['NAME']: PLASTIC_BOTTLE_BASE12,
    PLASTIC_BOTTLE_BASE13['NAME']: PLASTIC_BOTTLE_BASE13,
    PLASTIC_BOTTLE_BASE14['NAME']: PLASTIC_BOTTLE_BASE14,
    PLASTIC_BOTTLE_BASE15['NAME']: PLASTIC_BOTTLE_BASE15,
    PLASTIC_BOTTLE_BASE16['NAME']: PLASTIC_BOTTLE_BASE16,
    PLASTIC_BOTTLE_BASE17['NAME']: PLASTIC_BOTTLE_BASE17,
    PLASTIC_BOTTLE_BASE18['NAME']: PLASTIC_BOTTLE_BASE18,
    PLASTIC_BOTTLE_BASE19['NAME']: PLASTIC_BOTTLE_BASE19,
    PLASTIC_BOTTLE_BASE2['NAME']: PLASTIC_BOTTLE_BASE2,
    PLASTIC_BOTTLE_BASE20['NAME']: PLASTIC_BOTTLE_BASE20,
    PLASTIC_BOTTLE_BASE21['NAME']: PLASTIC_BOTTLE_BASE21,
    PLASTIC_BOTTLE_BASE22['NAME']: PLASTIC_BOTTLE_BASE22,
    PLASTIC_BOTTLE_BASE3['NAME']: PLASTIC_BOTTLE_BASE3,
    PLASTIC_BOTTLE_BASE4['NAME']: PLASTIC_BOTTLE_BASE4,
    PLASTIC_BOTTLE_BASE5['NAME']: PLASTIC_BOTTLE_BASE5,
    PLASTIC_BOTTLE_BASE6['NAME']: PLASTIC_BOTTLE_BASE6,
    PLASTIC_BOTTLE_BASE7['NAME']: PLASTIC_BOTTLE_BASE7,
    PLASTIC_BOTTLE_BASE8['NAME']: PLASTIC_BOTTLE_BASE8,
    PLASTIC_BOTTLE_BASE9['NAME']: PLASTIC_BOTTLE_BASE9,
    PLATE_BASE0['NAME']: PLATE_BASE0,
    ROLL_PAPER_BASE0['NAME']: ROLL_PAPER_BASE0,
    ROLL_PAPER_BASE1['NAME']: ROLL_PAPER_BASE1,
    ROLL_PAPER_BASE2['NAME']: ROLL_PAPER_BASE2,
    ROLL_PAPER_BASE3['NAME']: ROLL_PAPER_BASE3,
    SHAMPOO_BASE1['NAME']: SHAMPOO_BASE1,
    SHAMPOO_BASE2['NAME']: SHAMPOO_BASE2,
    SHAMPOO_BASE3['NAME']: SHAMPOO_BASE3,
    SHAMPOO_BASE4['NAME']: SHAMPOO_BASE4,
    SHAMPOO_BASE5['NAME']: SHAMPOO_BASE5,
    SHAMPOO_BASE6['NAME']: SHAMPOO_BASE6,
    SHAMPOO_BASE7['NAME']: SHAMPOO_BASE7,
    SHOE_BOX_BASE0['NAME']: SHOE_BOX_BASE0,
    STAPLER_BASE1['NAME']: STAPLER_BASE1,
    STAPLER_BASE2['NAME']: STAPLER_BASE2,
    STAPLER_BASE3['NAME']: STAPLER_BASE3,
    STAPLER_BASE4['NAME']: STAPLER_BASE4,
    STAPLER_BASE5['NAME']: STAPLER_BASE5,
    STAPLER_BASE6['NAME']: STAPLER_BASE6,
    STAPLER_BASE7['NAME']: STAPLER_BASE7,
    TABLE_TENNIS_BASE0['NAME']: TABLE_TENNIS_BASE0,
    TABLE_TENNIS_BASE1['NAME']: TABLE_TENNIS_BASE1,
    TISSUE_BOX_BASE0['NAME']: TISSUE_BOX_BASE0,
    TISSUE_BOX_BASE1['NAME']: TISSUE_BOX_BASE1,
    TISSUE_BOX_BASE2['NAME']: TISSUE_BOX_BASE2,
    TISSUE_BOX_BASE3['NAME']: TISSUE_BOX_BASE3,
    TISSUE_BOX_BASE4['NAME']: TISSUE_BOX_BASE4,
    TISSUE_BOX_BASE5['NAME']: TISSUE_BOX_BASE5,
    TISSUE_BOX_BASE6['NAME']: TISSUE_BOX_BASE6,
    TRAY_BASE0['NAME']: TRAY_BASE0,
    TRAY_BASE1['NAME']: TRAY_BASE1,
    TRAY_BASE2['NAME']: TRAY_BASE2,
    TRAY_BASE3['NAME']: TRAY_BASE3,
    WOODEN_BOX_BASE0['NAME']: WOODEN_BOX_BASE0
}
