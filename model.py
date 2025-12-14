import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual


class PrunedMobileNetV2(nn.Module):
    """
    CIFAR-style MobileNetV2 rebuilt from channel-level cfg.
    cfg example:
    [None, 8, 14, 14, 18, ..., 209, None]
    """

    def __init__(self, cfg, num_classes=10):
        super().__init__()

        # -------------------------------
        # Extract valid channels
        # -------------------------------
        channels = [c for c in cfg if c is not None]
        assert len(channels) == 17, "Expected 17 channels (1 stem + 16 blocks)"

        # -------------------------------
        # Stem
        # -------------------------------
        self.features = []
        input_channel = 3
        stem_out = channels[0]

        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, stem_out, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_out),
                nn.ReLU6(inplace=True),
            )
        )

        input_channel = stem_out
        ch_idx = 1

        # -------------------------------
        # CIFAR MobileNetV2 blocks (16)
        # (t, n, s)
        # -------------------------------
        block_cfg = [
            (6, 2, 1),
            (6, 3, 2),
            (6, 4, 2),
            (6, 3, 1),
            (6, 3, 2),
            (6, 1, 1),
        ]

        for t, n, s in block_cfg:
            for i in range(n):
                out_channel = channels[ch_idx]
                ch_idx += 1

                stride = s if i == 0 else 1

                self.features.append(
                    InvertedResidual(
                        inp=input_channel,
                        oup=out_channel,
                        stride=stride,
                        expand_ratio=t,
                    )
                )

                input_channel = out_channel

        self.features = nn.Sequential(*self.features)

        # -------------------------------
        # Classifier
        # -------------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_mobilenet_v2(num_classes=10, cfg=None):
    assert cfg is not None, "Structured channel pruning requires cfg"
    return PrunedMobileNetV2(cfg=cfg, num_classes=num_classes)
