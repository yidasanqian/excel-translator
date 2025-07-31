"""数据类型配置 - 可配置的数据类型推断."""

from typing import Dict
import pandas as pd


class DataTypeConfig:
    """可配置的数据类型推断."""

    def __init__(self, config: Dict = None):
        self.config = config

    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """推断列的数据类型."""
        types = {}
        for col in df.columns:
            column_data = df[col]
            if isinstance(column_data, pd.DataFrame):
                column_data = column_data.iloc[:, 0]
            values = column_data.dropna().astype(str)
            if values.empty:
                types[col] = "empty"
                continue
            try:
                values.astype(float)
                types[col] = "numeric"
                continue
            except ValueError:
                pass
            types[col] = "text"
        return types

    def get_config(self) -> Dict:
        """获取当前配置."""
        return self.config
