# runner.py (修改后)

import bilby
import numpy as np
from typing import Sequence, Tuple, Optional

from .types import ObsData, Setups, ModelParams, VegasMC, ParamDef, Scale
# We now need to pass the param_defs to the likelihood
from .likelihood import VegasGlowLikelihood 

class BilbyFitter:
    """
    使用 bilby 进行 afterglow 模型拟合的高级接口。
    """
    def __init__(self, data: ObsData, config: Setups):
        self.data = data
        self.config = config
        self._param_defs = None

    def _create_priors_from_defs(self, param_defs: Sequence[ParamDef]) -> bilby.core.prior.PriorDict:
        """
        一个辅助函数，将 ParamDef 列表转换为 bilby 的 PriorDict。
        对于 LOG 尺度的参数，我们将创建一个名为 log10_<name> 的新参数并使用 Uniform 先验。
        """
        priors = bilby.core.prior.PriorDict()

        for pd in param_defs:
            if pd.scale is Scale.LOG:
                # 1. 创建一个新的参数名称
                log_name = f"log10_{pd.name}"
                # 2. 对 log10(value) 使用线性均匀先验
                priors[log_name] = bilby.core.prior.Uniform(
                    minimum=np.log10(pd.lower), 
                    maximum=np.log10(pd.upper), 
                    name=log_name,
                    latex_label=f"$\\log_{{10}} {pd.latex or pd.name}$" # 让角图的标签更美观
                )
            elif pd.scale is Scale.LINEAR:
                priors[pd.name] = bilby.core.prior.Uniform(
                    minimum=pd.lower, maximum=pd.upper, name=pd.name, latex_label=f"${pd.latex or pd.name}$"
                )
            elif pd.scale is Scale.FIXED:
                val = 0.5 * (pd.lower + pd.upper)
                priors[pd.name] = bilby.core.prior.DeltaFunction(peak=val, name=pd.name, latex_label=f"${pd.latex or pd.name}$")

        return priors

    def fit(
        self,
        param_defs: Sequence[ParamDef],
        label: str = "vegasglow_fit",
        outdir: str = "output",
        **sampler_kwargs
    ) -> bilby.core.result.Result:
        """
        使用 bilby 运行 MCMC 采样器。
        """
        self._param_defs = list(param_defs)

        # 1. 根据 ParamDef 创建 bilby 先验
        priors = self._create_priors_from_defs(self._param_defs)

        # 2. 实例化我们的自定义似然函数
        # 我们将 param_defs 传递给似然函数，以便它知道哪些参数需要转换回来
        likelihood = VegasGlowLikelihood(self.data, self.config, self._param_defs)

        # 3. 调用 bilby 运行采样器！
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            label=label,
            outdir=outdir,
            **sampler_kwargs
        )

        result.plot_corner()
        return result

    # specific_flux 方法不需要改变，因为它接收一个普通字典，
    # 但你需要确保传入的字典键是 log10_<name> 形式。
    # 这里为了方便使用，我们可以稍微修改一下。

    def specific_flux(
        self,
        best_params: dict,
        t: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        """
        使用最佳拟合参数计算光变曲线。
        这个函数现在会自动处理 log10_<name> 形式的参数。
        """
        if self._param_defs is None:
            raise RuntimeError("必须先调用 .fit() 方法来设置参数定义。")

        # 使用 VegasGlowLikelihood 中的转换逻辑
        p = VegasGlowLikelihood.build_model_params(best_params, self._param_defs)
        
        model = VegasMC(self.data)
        model.set(self.config)
        return model.specific_flux(p, t, nu)
