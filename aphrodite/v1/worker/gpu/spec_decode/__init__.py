# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from aphrodite.config import AphroditeConfig

if TYPE_CHECKING:
    from aphrodite.v1.worker.gpu.states import RequestState


def init_speculator(
    aphrodite_config: AphroditeConfig,
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator for this config."""
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        from aphrodite.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(aphrodite_config, device)
    elif speculative_config.method == "dflash":
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from aphrodite.v1.worker.gpu.spec_decode.lilicorr.speculator import (
                LiLiCorrSpeculator,
            )

            return LiLiCorrSpeculator(aphrodite_config, device)
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from aphrodite.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(aphrodite_config, device)
        from aphrodite.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(aphrodite_config, device)
    elif speculative_config.method == "dspark":
        from aphrodite.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(aphrodite_config, device)
    elif speculative_config.use_gemma4_mtp():
        from aphrodite.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(aphrodite_config, device)
    elif speculative_config.use_multi_module_mtp():
        from aphrodite.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        return MultiModuleMTPSpeculator(aphrodite_config, device)
    elif speculative_config.method == "mtp":
        from aphrodite.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(aphrodite_config, device)
    elif speculative_config.use_eagle():
        from aphrodite.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(aphrodite_config, device)
    elif speculative_config.uses_draft_model():
        from aphrodite.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        return StandaloneARSpeculator(aphrodite_config, device)
    elif speculative_config.use_ngram():
        from aphrodite.v1.worker.gpu.spec_decode.ngram.speculator import (
            NgramGPUSpeculator,
        )

        return NgramGPUSpeculator(aphrodite_config, device, req_states)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
