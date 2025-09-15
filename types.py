# src/vegasglow/types.py
# types.py (修改后)

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Callable, Any, Optional
from enum import Enum
# 假设这些是你自定义的C++模块导入
from .VegasAfterglowC import ModelParams, Setups, ObsData, VegasMC, Ejecta, Wind, Medium, ISM, TophatJet, GaussianJet, PowerLawJet, TwoComponentJet, StepPowerLawJet, Radiation, Observer, Model, Magnetar


# FitResult 类已被移除，因为我们将使用 bilby.core.result.Result 对象


class Scale(Enum):
    """
    参数的尺度定义。
    """
    LINEAR = "linear"  # 线性采样
    LOG = "log"        # 对数采样 (log10)
    FIXED = "fixed"      # 固定参数，不采样


@dataclass
class ParamDef:
    """
    用于MCMC的单个参数定义。
    scale=LOG 表示我们采样 log10(x)，然后通过 10**v 转换回原值。
    scale=FIXED 表示此参数是固定的，不会在采样器中出现。
    """
    name:   str      # 参数名称，必须与 ModelParams 中的属性名匹配
    lower:  float    # 参数下界
    upper:  float    # 参数上界
    scale:  Scale = Scale.LINEAR
    latex:  Optional[str] = None  # 用于绘图的 LaTeX 标签
