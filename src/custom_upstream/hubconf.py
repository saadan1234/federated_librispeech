"""
Hub configuration for custom HuBERT upstream model.
This registers the CustomHubertExpert with s3prl.
"""

from .expert import CustomHubertExpert as _CustomHubertExpert


def custom_hubert_local(*args, **kwargs):
    """
    Custom HuBERT upstream that loads federated pretrained checkpoints.

    Usage:
        -u custom_hubert_local -k /path/to/checkpoint.pt

    Args:
        ckpt: Path to the federated pretraining checkpoint
        model_config: Optional model configuration file (not used in this implementation)
    """
    return _CustomHubertExpert(*args, **kwargs)


def custom_hubert_url(*args, **kwargs):
    """
    Custom HuBERT upstream for URL-based checkpoints.
    """
    return _CustomHubertExpert(*args, **kwargs)


def custom_hubert(*args, **kwargs):
    """
    Default custom HuBERT upstream.
    """
    return _CustomHubertExpert(*args, **kwargs)
