from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
import os

DATA_DIR = os.environ.get("UDFS_DATA_DIR", "/public/home/wzx2003/stock_dataset")
OPEN = 'open'
CLOSE = 'close'
HIGH = 'high'
LOW = 'low'
VOLUME = 'volume'

class Stock_Data():
    def __init__(self, stock_list=['000001.XSHE', '000002.XSHE'],
                 k_line='1d', start_date='', end_date=''):
        """
        初始化股票数据集

        Args:
            stock_list: 股票代码列表
            k_line: K线类型，默认'1d'（日线）
            start_date: 开始日期
            end_date: 结束日期
        """
        data_list = []
        if stock_list == 'csi300' or stock_list == 'csi500':
            stock_list = pd.read_csv(f"{DATA_DIR}/{stock_list}_constituents.csv")['stock_code'].tolist()
        stock_list = sorted(stock_list)  # 确保股票代码有序，方便后续处理
        for stock_code in stock_list:
            stock_file = f"{DATA_DIR}/{k_line}/{stock_code}/{stock_code}.csv"
            if os.path.exists(stock_file):
                print(f"Loading data from {stock_code}...")
                data = pd.read_csv(stock_file, encoding="utf-8")

                # 统一列名
                if 'order_book_id' in data.columns:
                    data = data.rename(columns={'order_book_id': 'stock_code'})
                if 'stock_code' not in data.columns:
                    data['stock_code'] = stock_code

                # 确保有日期列
                if 'datetime' in data.columns:
                    data.rename(columns={'datetime': 'date'}, inplace=True)
                elif 'date' not in data.columns:
                    raise ValueError("DataFrame must contain 'datetime' or 'date' column.")

                # 转换日期格式
                data['date'] = pd.to_datetime(data['date'])

                # 日期过滤
                if start_date != '':
                    data = data[data['date'] >= start_date]
                if end_date != '':
                    data = data[data['date'] <= end_date]

                data_list.append(data)
            else:
                print(f"Data for {stock_code} not found!")
                continue

        self.data = pd.concat(data_list, axis=0, ignore_index=True)
        self.id_slice, self.daily_slice = self.get_index_ranges()
        self.test_slice = self.daily_slice
        self.n_days = len(self.daily_slice)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_index_ranges(self) -> Tuple[List[slice], List[slice]]:
        """
        获取股票种类索引和日线索引

        Returns:
            stock_slice: 按股票代码分组的索引切片列表
            daily_slice: 按日期分组的索引切片列表
        """
        if 'stock_code' not in self.data.columns:
            raise ValueError("The 'stock_code' column is missing in the data.")
        if 'date' not in self.data.columns:
            raise ValueError("The 'date' column is missing in the data.")

        stock_ranges = []
        daily_ranges = []

        # 1. 按股票代码分组，生成股票种类索引
        grouped_by_stock = self.data.groupby('stock_code')
        for stock_code, group in grouped_by_stock:
            start_idx = group.index[0]
            end_idx = group.index[-1] + 1  # 右开区间
            stock_ranges.append(slice(start_idx, end_idx))

        # 2. 按日期分组，生成日线索引
        grouped_by_date = self.data.groupby('date')
        for date, group in grouped_by_date:
            #print(date,group.index)
            start_idx = group.index[0]
            end_idx = group.index[-1] + 1  # 右开区间
            daily_ranges.append(group.index)
            #print(daily_ranges)

        return stock_ranges, daily_ranges

if __name__ == "__main__":
    stock_data = Stock_Data(stock_list=['600000.XSHG','000001.XSHE','000002.XSHE'], k_line='1d',start_date='2020-01-01', end_date='2020-01-10')
    for index in stock_data.id_slice:
        print(stock_data.data.iloc[index])
    for index in stock_data.daily_slice:
        print(stock_data.data.iloc[index])