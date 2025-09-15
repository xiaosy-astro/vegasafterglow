# src/vegasglow/likelihood.py (新文件)

import bilby
import numpy as np
from typing import Sequence, Dict

from .types import ObsData, Setups, ModelParams, VegasMC, ParamDef, Scale

class VegasGlowLikelihood(bilby.likelihood.Likelihood):
    """
    VegasGlow 模型的 bilby 似然函数。

    这个类负责将 bilby 采样器提供的参数（包括 log10 形式的参数）
    转换为物理模型（VegasMC）所需的线性参数。
    """
    def __init__(self, data: ObsData, config: Setups, param_defs: Sequence[ParamDef]):
        """
        初始化似然函数。

        参数
        ----------
        data: 观测数据
        config: 模型配置
        param_defs: 参数定义，用于识别哪些参数需要从 log10 转换回来
        """
        self.data = data
        self.config = config
        self.param_defs = param_defs
        
        # 从 param_defs 中提取参数名，以供 bilby 使用
        # 注意：这里我们使用的是转换后的名字 (log10_<name>)
        param_keys = []
        for pd in self.param_defs:
            if pd.scale == Scale.LOG:
                param_keys.append(f"log10_{pd.name}")
            elif pd.scale == Scale.LINEAR:
                param_keys.append(pd.name)
        
        super().__init__(parameters={key: None for key in param_keys})
        
        # 创建一个可重用的模型实例
        self.model = VegasMC(self.data)
        self.model.set(self.config)

    @staticmethod
    def build_model_params(params_dict: Dict, param_defs: Sequence[ParamDef]) -> ModelParams:
        """
        一个静态辅助方法，将 bilby 提供的参数字典转换为 ModelParams 对象。
        这里执行从 log10 到线性的逆转换。
        """
        p = ModelParams()
        
        # 先填充固定参数的默认值
        for pd in param_defs:
            if pd.scale == Scale.FIXED:
                setattr(p, pd.name, 0.5 * (pd.lower + pd.upper))

        # 遍历从 bilby 接收到的采样参数
        for name, value in params_dict.items():
            if name.startswith("log10_"):
                # 这是我们转换过的参数
                original_name = name[6:] # 移除 "log10_" 前缀
                linear_value = 10**value # ⭐ 在这里执行逆转换！
                if hasattr(p, original_name):
                    setattr(p, original_name, linear_value)
            else:
                # 这是普通的线性参数
                if hasattr(p, name):
                    setattr(p, name, value)
        return p

    def log_likelihood(self) -> float:
        """
        计算对数似然。
        """
        # 1. 从 self.parameters (由 bilby 填充) 构建 ModelParams
        # self.parameters 是一个字典，例如 {'log10_E_iso': 52.1, 'theta_c': 0.1, ...}
        model_params = self.build_model_params(self.parameters, self.param_defs)

        # 2. 使用转换后的物理参数计算 chi2
        try:
            chi2 = self.model.estimate_chi2(model_params)
            return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
        except Exception:
            # 捕获任何 C++ model 可能抛出的异常
            return -np.inf
