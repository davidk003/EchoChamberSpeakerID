"""Export CAM++ to a fixed-shape ONNX graph for the Qualcomm QNN HTP (NPU) backend.

Run this **once**, in the x64 driver venv built by ``setup_speakerid_qnn.py``
(has torch, onnx, onnxruntime, onnxsim, huggingface_hub)::

    .venv-speakerid-x64/Scripts/python.exe scripts/export_speakerid_qnn.py

The QNN HTP backend requires every tensor shape to be static at compile time
(no dynamic axes), so the input is a fixed ``(1, T_MAX_FRAMES, 80)`` fbank
tensor rather than a variable-length one. See
``echochamber/speakerid/qnn_driver_worker.py`` for the padding/truncation
logic applied at inference time.

**Why the model architecture lives here and nowhere else.**  This is the only
place in the whole project that needs the CAM++ *model graph* itself, as
opposed to feature extraction
(:mod:`echochamber.speakerid.campplus`) or ONNX Runtime inference
(``echochamber/speakerid/qnn_npu_worker.py``). Everything downstream of this
script only ever runs the already-exported ``.onnx`` file, so keeping the
``torch.nn.Module`` definition -- and its hard dependency on ``torch`` -- out
of the runtime package means the deployed app never needs to import it.
Ported unchanged from the sibling ``cam-script`` repository's
``campplus.py``/``campplus_qnn_export.py``, which are not pip-installable and
therefore cannot be runtime dependencies of this project.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echochamber.speakerid.campplus import FEAT_DIM  # noqa: E402

EMBEDDING_SIZE = 192
# Covers a comfortable margin over a multi-second wake-phrase clip.
# min_samples(T) = 400 + (T - 1) * 160 samples at 16 kHz.
T_MAX_FRAMES = 600

_REPO = "funasr/campplus"
_WEIGHTS = "campplus_cn_common.bin"

ONNX_PATH = "models/speakerid/campplus_qnn.onnx"
"""Must match ``echochamber.speakerid.config._ONNX_DIR`` / ``_ONNX_NAME``."""


def _build_model(embedding_size: int = EMBEDDING_SIZE) -> Any:
    """Construct the CAM++ ``nn.Module`` graph.  Identical to the sibling
    ``cam-script`` repository's ``campplus.py`` architecture, needed here only
    to trace an export -- see the module docstring for why this is the one
    place in the project that builds it.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812
    import torch.utils.checkpoint as cp
    from torch import nn

    def get_nonlinear(config_str: str, channels: int) -> Any:
        nonlinear = nn.Sequential()
        for name in config_str.split("-"):
            if name == "relu":
                nonlinear.add_module("relu", nn.ReLU(inplace=True))
            elif name == "prelu":
                nonlinear.add_module("prelu", nn.PReLU(channels))
            elif name == "batchnorm":
                nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
            elif name == "batchnorm_":
                nonlinear.add_module(
                    "batchnorm", nn.BatchNorm1d(channels, affine=False)
                )
            else:
                raise ValueError(f"Unexpected module ({name}).")
        return nonlinear

    def statistics_pooling(
        x: Any, dim: int = -1, keepdim: bool = False, unbiased: bool = True
    ) -> Any:
        mean = x.mean(dim=dim)
        std = x.std(dim=dim, unbiased=unbiased)
        stats = torch.cat([mean, std], dim=-1)
        if keepdim:
            stats = stats.unsqueeze(dim=dim)
        return stats

    class StatsPool(nn.Module):
        def forward(self, x: Any) -> Any:
            return statistics_pooling(x)

    class TDNNLayer(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = False,
            config_str: str = "batchnorm-relu",
        ) -> None:
            super().__init__()
            if padding < 0:
                assert kernel_size % 2 == 1, (
                    f"Expect equal paddings, but got even kernel size "
                    f"({kernel_size})"
                )
                padding = (kernel_size - 1) // 2 * dilation
            self.linear = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.nonlinear = get_nonlinear(config_str, out_channels)

        def forward(self, x: Any) -> Any:
            x = self.linear(x)
            x = self.nonlinear(x)
            return x

    class CAMLayer(nn.Module):
        def __init__(
            self,
            bn_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            dilation: int,
            bias: bool,
            reduction: int = 2,
        ) -> None:
            super().__init__()
            self.linear_local = nn.Conv1d(
                bn_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: Any) -> Any:
            y = self.linear_local(x)
            context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
            context = self.relu(self.linear1(context))
            m = self.sigmoid(self.linear2(context))
            return y * m

        def seg_pooling(
            self, x: Any, seg_len: int = 100, stype: str = "avg"
        ) -> Any:
            if stype == "avg":
                seg = F.avg_pool1d(
                    x, kernel_size=seg_len, stride=seg_len, ceil_mode=True
                )
            elif stype == "max":
                seg = F.max_pool1d(
                    x, kernel_size=seg_len, stride=seg_len, ceil_mode=True
                )
            else:
                raise ValueError("Wrong segment pooling type.")
            shape = seg.shape
            seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
            seg = seg[..., : x.shape[-1]]
            return seg

    class CAMDenseTDNNLayer(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bn_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = False,
            config_str: str = "batchnorm-relu",
            memory_efficient: bool = False,
        ) -> None:
            super().__init__()
            assert kernel_size % 2 == 1, (
                f"Expect equal paddings, but got even kernel size ({kernel_size})"
            )
            padding = (kernel_size - 1) // 2 * dilation
            self.memory_efficient = memory_efficient
            self.nonlinear1 = get_nonlinear(config_str, in_channels)
            self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
            self.nonlinear2 = get_nonlinear(config_str, bn_channels)
            self.cam_layer = CAMLayer(
                bn_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )

        def bn_function(self, x: Any) -> Any:
            return self.linear1(self.nonlinear1(x))

        def forward(self, x: Any) -> Any:
            if self.training and self.memory_efficient:
                x = cp.checkpoint(self.bn_function, x)
            else:
                x = self.bn_function(x)
            x = self.cam_layer(self.nonlinear2(x))
            return x

    class CAMDenseTDNNBlock(nn.ModuleList):
        def __init__(
            self,
            num_layers: int,
            in_channels: int,
            out_channels: int,
            bn_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = False,
            config_str: str = "batchnorm-relu",
            memory_efficient: bool = False,
        ) -> None:
            super().__init__()
            for i in range(num_layers):
                layer = CAMDenseTDNNLayer(
                    in_channels=in_channels + i * out_channels,
                    out_channels=out_channels,
                    bn_channels=bn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                )
                self.add_module("tdnnd%d" % (i + 1), layer)

        def forward(self, x: Any) -> Any:
            for layer in self:
                x = torch.cat([x, layer(x)], dim=1)
            return x

    class TransitLayer(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            config_str: str = "batchnorm-relu",
        ) -> None:
            super().__init__()
            self.nonlinear = get_nonlinear(config_str, in_channels)
            self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

        def forward(self, x: Any) -> Any:
            x = self.nonlinear(x)
            x = self.linear(x)
            return x

    class DenseLayer(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            config_str: str = "batchnorm-relu",
        ) -> None:
            super().__init__()
            self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
            self.nonlinear = get_nonlinear(config_str, out_channels)

        def forward(self, x: Any) -> Any:
            if len(x.shape) == 2:
                x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
            else:
                x = self.linear(x)
            x = self.nonlinear(x)
            return x

    class BasicResBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=(stride, 1),
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=(stride, 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

        def forward(self, x: Any) -> Any:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class FCM(nn.Module):
        def __init__(
            self,
            block: Any = BasicResBlock,
            num_blocks: tuple[int, int] = (2, 2),
            m_channels: int = 32,
            feat_dim: int = 80,
        ) -> None:
            super().__init__()
            self.in_planes = m_channels
            self.conv1 = nn.Conv2d(
                1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(m_channels)

            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
            self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

            self.conv2 = nn.Conv2d(
                m_channels,
                m_channels,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(m_channels)
            self.out_channels = m_channels * (feat_dim // 8)

        def _make_layer(
            self, block: Any, planes: int, num_blocks: int, stride: int
        ) -> Any:
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, planes, s))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            x = x.unsqueeze(1)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = F.relu(self.bn2(self.conv2(out)))

            shape = out.shape
            out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
            return out

    class CAMPPlus(nn.Module):
        def __init__(
            self,
            feat_dim: int = FEAT_DIM,
            embedding_size: int = EMBEDDING_SIZE,
            growth_rate: int = 32,
            bn_size: int = 4,
            init_channels: int = 128,
            config_str: str = "batchnorm-relu",
            memory_efficient: bool = True,
        ) -> None:
            super().__init__()

            self.head = FCM(feat_dim=feat_dim)
            channels = self.head.out_channels

            self.xvector = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "tdnn",
                            TDNNLayer(
                                channels,
                                init_channels,
                                5,
                                stride=2,
                                dilation=1,
                                padding=-1,
                                config_str=config_str,
                            ),
                        ),
                    ]
                )
            )
            channels = init_channels
            for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
            ):
                block = CAMDenseTDNNBlock(
                    num_layers=num_layers,
                    in_channels=channels,
                    out_channels=growth_rate,
                    bn_channels=bn_size * growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                )
                self.xvector.add_module("block%d" % (i + 1), block)
                channels = channels + num_layers * growth_rate
                self.xvector.add_module(
                    "transit%d" % (i + 1),
                    TransitLayer(
                        channels, channels // 2, bias=False, config_str=config_str
                    ),
                )
                channels //= 2

            self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )

            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: Any) -> Any:
            x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
            x = self.head(x)
            x = self.xvector(x)
            return x

    return CAMPPlus(feat_dim=FEAT_DIM, embedding_size=embedding_size)


def _download_weights() -> str:
    """Return a local path to ``campplus_cn_common.bin``, fetching it if needed."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_REPO, filename=_WEIGHTS)


def _load_checkpoint(path: Path) -> tuple[Any, int]:
    """Load a weights file, returning ``(state_dict, embedding_size)``."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"], raw.get("embedding_size", EMBEDDING_SIZE)
    return raw, EMBEDDING_SIZE


def export_onnx(
    weights: str | None = None, onnx_path: str = ONNX_PATH, t_max: int = T_MAX_FRAMES
) -> None:
    """Export CAM++ to a static-shape ONNX graph and sanity-check it.

    Args:
        weights: Path to ``campplus_cn_common.bin`` (or a finetuned
            checkpoint), or ``None`` to download the base model.
        onnx_path: Where to write the exported graph.
        t_max: Fixed fbank-frame length the exported graph accepts.
    """
    import onnx
    import onnxruntime
    import torch
    from onnxsim import simplify

    path = Path(weights) if weights else Path(_download_weights())
    state, embedding_size = _load_checkpoint(path)
    model = _build_model(embedding_size)
    model.load_state_dict(state, strict=True)
    model.eval()

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, t_max, FEAT_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["fbank"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # no dynamic_axes at all -> every dim baked in as static
    )
    print(f"exported {onnx_path}")

    # torch's exporter emits a Shape/Gather/Concat chain for
    # CAMLayer.seg_pooling's expand() call instead of folding it into a
    # constant, even though every shape is static at trace time. QNN EP
    # requires Expand's target shape to be a Constant/initializer, so
    # constant-fold the graph with onnx-simplifier before checking/using it.
    simplified, ok = simplify(onnx.load(onnx_path))
    if not ok:
        raise RuntimeError("onnx-simplifier failed to validate the simplified graph")
    onnx.save(simplified, onnx_path)
    print(f"simplified {onnx_path}")

    _check_static_shapes(onnx, onnx_path)
    _check_parity(onnxruntime, torch, model, dummy, onnx_path)


def _check_static_shapes(onnx: Any, onnx_path: str) -> None:
    """Fail loudly if any dim is symbolic, or if Expand's shape isn't constant."""
    model = onnx.load(onnx_path)
    graph = model.graph

    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = vi.type.tensor_type.shape
        for dim in shape.dim:
            if dim.HasField("dim_param") and dim.dim_param:
                raise RuntimeError(
                    f"dynamic dim {dim.dim_param!r} found on {vi.name!r} -- "
                    "QNN HTP requires fully static shapes"
                )

    constants = {n.output[0] for n in graph.node if n.op_type == "Constant"}
    initializers = {i.name for i in graph.initializer}
    static_producers = constants | initializers

    bad_expands = [
        n.name or n.output[0]
        for n in graph.node
        if n.op_type == "Expand" and n.input[1] not in static_producers
    ]
    if bad_expands:
        raise RuntimeError(
            f"Expand node(s) with a non-constant shape input: {bad_expands} -- "
            "QNN EP requires Expand's target shape to be a Constant/initializer"
        )

    print(f"static shape check passed: {len(graph.node)} nodes, no dynamic dims")


def _check_parity(
    onnxruntime: Any, torch: Any, model: Any, dummy: Any, onnx_path: str
) -> None:
    """torch vs onnxruntime-CPU on the exported graph: must match almost exactly."""
    import numpy as np

    with torch.no_grad():
        torch_out = model(dummy)[0].numpy()

    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"fbank": dummy.numpy()})
    ort_out = ort_out[0]

    cos = float(
        np.dot(torch_out, ort_out) / (np.linalg.norm(torch_out) * np.linalg.norm(ort_out))
    )
    print(f"torch vs onnxruntime-CPU cosine similarity: {cos:.6f}")
    if cos < 0.9999:
        raise RuntimeError(f"export parity check failed: cosine={cos:.6f} < 0.9999")


def main() -> None:
    p = argparse.ArgumentParser(description="Export CAM++ to a static-shape ONNX graph")
    p.add_argument("--weights", help="path to campplus_cn_common.bin")
    p.add_argument("--onnx-path", default=ONNX_PATH)
    p.add_argument("--t-max", type=int, default=T_MAX_FRAMES)
    args = p.parse_args()

    if args.weights and not Path(args.weights).is_file():
        sys.exit(f"error: no such weights file: {args.weights}")

    export_onnx(weights=args.weights, onnx_path=args.onnx_path, t_max=args.t_max)


if __name__ == "__main__":
    main()
