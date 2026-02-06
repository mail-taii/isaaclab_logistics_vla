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

CRACKER_BOX_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_003_cracker_box/003_cracker_box.usd",
    'X_LENGTH':0.16,
    'Y_LENGTH':0.20,
    'Z_LENGTH':0.06,
    'SPARSE_ORIENT':(0,90,0)    
}

SUGER_BOX_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_004_sugar_box/004_sugar_box.usd",
    'X_LENGTH':0.09,
    'Y_LENGTH':0.17,
    'Z_LENGTH':0.04,
    'SPARSE_ORIENT':(0,90,0)
}

TOMATO_SOUP_CAN_PARAMS = {
    'USD_PATH':f"{ASSET_ROOT_PATH}/props/Collected_005_tomato_soup_can/005_tomato_soup_can.usd",
    'RADIUS':0.035,
    'X_LENGTH':0.07,
    'Y_LENGTH':0.10,
    'Z_LENGTH':0.07,
    'SPARSE_ORIENT':(90,0,0)
}