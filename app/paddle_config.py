"""PaddlePaddle configuration - MUST be imported before any paddle imports.

This module sets environment variables to disable OneDNN/MKL-DNN which
causes compatibility issues on Windows with certain PaddlePaddle versions.

The "OneDnnContext does not have the input Filter" error occurs when
PaddlePaddle tries to use OneDNN optimizations that aren't compatible
with the current system configuration.
"""

import os

# =============================================================================
# ENVIRONMENT VARIABLES - Set before any paddle imports
# =============================================================================

# Disable OneDNN/MKL-DNN completely
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["MKLDNN_VERBOSE"] = "0"
os.environ["PADDLE_WITH_MKLDNN"] = "OFF"

# Disable MKL-DNN cache
os.environ["FLAGS_mkldnn_cache_capacity"] = "0"

# Force CPU execution without hardware-specific optimizations
os.environ["FLAGS_use_gpu"] = "0"
os.environ["FLAGS_use_xpu"] = "0"
os.environ["FLAGS_use_npu"] = "0"

# Disable fused operations that may use OneDNN
os.environ["FLAGS_enable_fused_conv2d"] = "0"
os.environ["FLAGS_fuse_relu_depthwise_conv"] = "0"

# =============================================================================
# PROGRAMMATIC CONFIGURATION - Applied after paddle import
# =============================================================================


def configure_paddle() -> None:
    """Configure PaddlePaddle runtime settings.

    This function should be called after paddle is imported to ensure
    all runtime settings are properly applied.
    """
    try:
        import paddle

        # Set device to CPU explicitly
        paddle.device.set_device("cpu")

        # Disable MKL-DNN via paddle flags if available
        if hasattr(paddle, "set_flags"):
            paddle.set_flags(
                {
                    "FLAGS_use_mkldnn": False,
                }
            )
    except ImportError:
        pass  # paddle not yet installed
    except Exception:
        pass  # Ignore errors during configuration
