import sys
sys.path.append('.')
from expr_cal.stock_data import Stock_Data
from DAG_search import config
from expr_cal.expression import *
import warnings
import re
import pandas as pd
from torch_utils.correlation import *
from torch_utils.pytorch_utils import *
warnings.simplefilter("ignore")
def wrap_constants(expr: str):
    """
    将表达式中的数值常数封装为 Constant('原始常数')
    """
    pattern = re.compile(
        r'(?<![A-Za-z_])'     # 前面不是字母或 _
        r'(\d+(\.\d+)?)'      # 数字：整数或小数
        r'(?![A-Za-z_])'      # 后面不是字母或 _
    )

    return pattern.sub(lambda m: f"Constant('{m.group(1)}')", expr)

def add_mean_window(expr: str, target='mean', window=5):
    
    while expr.find(target) != -1:
        i = expr.find(target)
        start = i + len(target) +1 
        depth = 1
        j = start
        
        # 括号匹配
        while j < len(expr) and depth > 0:
            if expr[j] == "(":
                depth += 1
            elif expr[j] == ")":
                depth -= 1
            j += 1
        
        if depth != 0:
            raise ValueError("括号不匹配")
        
        # expr[i:j] 是完整的 MEAN(expr)
        expr = expr[:i] + f"{config.ROLLING_DICT[target]}({expr[start:j-1]},{window})" + expr[j:]

    return expr


def transform(expr:str,c: list = None):
    for i in config.feature_dict.keys():
        expr = expr.replace(i,f'Feature("{config.feature_dict[i]}")')
    
    expr = wrap_constants(expr)

    for i in config.ROLLING_DICT.keys():
        expr = add_mean_window(expr, target=i)
    
    
    return expr

if __name__=="__main__":
    expr1 = transform('min(x_4)')
    expr2 = transform('1/(x_4*x_0)/(x_4*x_0)')
    stock_list = ['600000.XSHG', '000001.XSHE', '000002.XSHE']
    data = Stock_Data(stock_list=stock_list, k_line='1d', start_date='2010-01-01',end_date='2018-12-31')
    target = ((Ref(Feature('close') , -2)/Ref(Feature('close'), -1) - 1)*1000).evaluate(data).fillna(0).values
    final_result = []
    #print(Feature("volume").evaluate(data))
    try:
        value = (eval(expr1).evaluate(data)).values
        invalues = np.isinf(value) | np.isnan(value)
        value[invalues] = 0
        final_result.append(normalize_by_day(value,data))
    except Exception as e:
        final_result.append(Constant(0).evaluate(data).values)

    try:
        value = (eval(expr2).evaluate(data)).values
        invalues = np.isinf(value) | np.isnan(value)
        value[invalues] = 0
        final_result.append(normalize_by_day(value,data))
    except Exception as e:
        final_result.append(Constant(0).evaluate(data).values)
    
    target = normalize_by_day(target, data)
    value1 = final_result[0]
    value2 = final_result[1]
    if len(target.shape) == 1:
        target = target.reshape(1,-1)
    
    if data.n_days == 0:
        print("No data loaded, score = 0")
        score = 0
    else:
        score = np.dot(target.T, final_result[0]) / data.n_days
    print(score.item())