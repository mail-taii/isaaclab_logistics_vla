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
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_003_cracker_box/003_cracker_box.usd",
    'X_LENGTH':0.16,
    'Y_LENGTH':0.20,
    'Z_LENGTH':0.06,
    'SPARSE_ORIENT':(0,90,0),   #相对于箱子的坐标
    "DENSE_ORIENT":[(0,90,0),(0,0,0)],
    'STACK_ORIENT':(0,0,0)         # Z最小，默认朝向即可
}

SUGER_BOX_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_004_sugar_box/004_sugar_box.usd",
    'X_LENGTH':0.09,
    'Y_LENGTH':0.17,
    'Z_LENGTH':0.04,
    'SPARSE_ORIENT':(0,90,0),
    "DENSE_ORIENT":[(0,90,0),(0,0,0)],
    'STACK_ORIENT':(0,0,0)         # Z最小，默认朝向即可
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
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_SFSmall/SFSmall.usdc",
    "RADIUS":0.035,
    'X_LENGTH':0.34,    
    'Y_LENGTH':0.43,
    'Z_LENGTH':0.08,
    'SPARSE_ORIENT':(0,0,0),#or (0,0,0)
    'STACK_ORIENT':(0,0,0),
    "DENSE_ORIENT":[(0,0,0)]
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
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_SFBig/SFBig.usdc",
    "RADIUS":0.22,
    'X_LENGTH':0.47,    
    'Z_LENGTH':0.34,
    'Y_LENGTH':0.15,
    'SPARSE_ORIENT':(90,0,0), # (0,0,0)
    'STACK_ORIENT':(90,0,90),
    "DENSE_ORIENT":[(90,0,0),(0,0,0)]
}

PLASTIC_PACKAGE_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_plastic_package/plastic_package.usdc",
    'X_LENGTH':0.34,
    'Y_LENGTH':0.39,
    'Z_LENGTH':0.07,
    'SPARSE_ORIENT':(0,0,0),
    'STACK_ORIENT':(0,0,0)         # Z最小，默认朝向即可
}