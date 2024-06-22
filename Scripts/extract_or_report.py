import sys
import os
import re
from typing import List

def get_run_config(log_dir:str) -> List[str]:
    base_dir = os.path.basename(log_dir)
    items = base_dir.split('_')
    util = items[1]
    gp_pad = items[3]
    dp_pad = items[5]
    dpo = items[7]
    return [util, gp_pad, dp_pad, dpo]

def convert_runtime_string_to_second(runtime_string:str) -> float:
    items = runtime_string.split(':')
    if len(items) > 3:
        print(f"Error: {runtime_string}")
        return -1.0
    
    if len(items) == 3:
        hours = int(items[0])
        minutes = int(items[1])
        seconds = float(items[2])
        return hours*3600 + minutes*60 + seconds
    elif len(items) == 2:
        minutes = int(items[0])
        seconds = float(items[1])
        return minutes*60 + seconds
    
    return float(items[0])
    

def get_gplace_details(log_dir:str) -> List[float]:
    gplace_log = f"{log_dir}/3_3_place_gp.log"
    dbu = 1000
    hpwl = -1.0
    run_time = -1.0
    if not os.path.exists(gplace_log):
        return [hpwl, run_time]
    fp = open(gplace_log, "r")
    lines = fp.readlines()
    fp.close()
    for line in lines:
        items = line.split()
        if len(items) == 4 and items[2] == 'DBU:':
            dbu = int(items[3])
        elif len(items) == 7 and items[0] == '[NesterovSolve]':
            hpwl = float(items[6])/dbu
        
    if lines[-1].startswith('Elapsed time'):
        runtime_string = lines[-1].split()[2]
        # Find runtime from this format: 0:47.58[h:]min:sec
        # Find the first match of a regex
        time_string = re.search(r"^([0-9:.]+)", runtime_string).group(1)
        run_time = convert_runtime_string_to_second(time_string)
    return [hpwl, run_time]

def get_dplace_details(log_dir:str) -> List[float]:
    dp_log = f"{log_dir}/3_5_place_dp.log"
    dp_hpwl = -1.0
    dp_runtime = -1.0
    if not os.path.exists(dp_log):
        return [dp_hpwl, dp_runtime]
    fp = open(dp_log, "r")
    lines = fp.readlines()
    fp.close()
    
    for line in lines:
        if re.search(" HPWL after", line):
            items = line.split()
            dp_hpwl = float(items[4])
        elif line.startswith('Elapsed time'):
            runtime_string = line.split()[2]
            time_string = re.search(r"^([0-9:.]+)", runtime_string).group(1)
            dp_runtime = convert_runtime_string_to_second(time_string)
    return [dp_hpwl, dp_runtime]

def get_route_report(log_dir:str) -> List[float]:
    route_log = f"{log_dir}/5_3_route.log"
    rwl = -1.0
    runtime = -1.0
    if not os.path.exists(route_log):
        return [rwl, runtime]
    fp = open(route_log, "r")
    lines = fp.readlines()
    fp.close()
    for line in lines:
        if line.startswith('Total wire length ='):
            items = line.split()
            rwl = float(items[4])
        elif line.startswith('Elapsed time'):
            runtime_string = line.split()[2]
            time_string = re.search(r"^([0-9:.]+)", runtime_string).group(1)
            runtime = convert_runtime_string_to_second(time_string)
    return [rwl, runtime]

def report_final_ppa(log_dir:str) -> List[float]:
    report_log = f"{log_dir}/6_report.log"
    wns =  float('-inf')
    tns =  float('-inf')
    power = -1.0
    if not os.path.exists(report_log):
        return [wns, tns, power]
    fp = open(report_log, "r")
    lines = fp.readlines()
    fp.close()
    
    for line in lines:
        if line.startswith('wns'):
            items = line.split()
            wns = float(items[1])
        elif line.startswith('tns'):
            items = line.split()
            tns = float(items[1])
        elif line.startswith('Total'):
            items = line.split()
            power = float(items[-2])
    return [wns, tns, power]

def extract_report(log_dir:str) -> None:
    # [util, gp_pad, dp_pad, dpo] = get_run_config(log_dir)
    [hpwl, gp_runtime] = get_gplace_details(log_dir)
    [dp_hpwl, dp_runtime] = get_dplace_details(log_dir)
    [rwl, route_runtime] = get_route_report(log_dir)
    [wns, tns, power] = report_final_ppa(log_dir)
    # print(f"{util},{gp_pad},{dp_pad},{dpo},{hpwl},{gp_runtime},{dp_hpwl},{dp_runtime},{rwl},{route_runtime},{wns},{tns},{power}")
    print(f"{hpwl},{gp_runtime},{dp_hpwl},{dp_runtime},{rwl},{route_runtime},{wns},{tns},{power}")

def extract_report_all(log_dir:str) -> None:
    # print("Util,GP_PAD,DP_PAD,DPO,GP_HPWL,GP_RUNTIME,DP_HPWL,DP_RUNTIME,RWL,ROUTE_RUNTIME,WNS,TNS,POWER")
    print("GP_HPWL,GP_RUNTIME,DP_HPWL,DP_RUNTIME,RWL,ROUTE_RUNTIME,WNS,TNS,POWER")
    extract_report(log_dir)

if __name__ == '__main__':
    log_dir = sys.argv[1]
    extract_report_all(log_dir)
    