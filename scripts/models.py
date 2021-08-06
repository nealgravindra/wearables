import torch
from torch import nn
import torch.nn.functional as F

####
# REF:
## 1. official implementation of InceptionTime (in tf):
##      https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
####

####
# InceptionTime dev
####
from typing import cast, Union, List

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class InceptionTime(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes
        }

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
                                                                          num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
        )

        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        return self.linear(x)


# class InceptionBlock(nn.Module):
#     """An inception block consists of an (optional) bottleneck, followed
#     by 3 conv1d layers. Optionally residual
#     """
#
#     def __init__(self, in_channels: int, out_channels: int,
#                  residual: bool, stride: int = 1, bottleneck_channels: int = 32,
#                  kernel_size: int = 41) -> None:
#         assert kernel_size > 3, "Kernel size must be strictly greater than 3"
#         super().__init__()
#
#         self.use_bottleneck = bottleneck_channels > 0
#         if self.use_bottleneck:
#             self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
#                                                 kernel_size=1, bias=False)
#         kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
#         start_channels = bottleneck_channels if self.use_bottleneck else in_channels
#         channels = [start_channels] + [out_channels] * 3
#         self.conv_layers = nn.Sequential(*[
#             Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
#                               kernel_size=kernel_size_s[i], stride=stride, bias=False)
#             for i in range(len(kernel_size_s))
#         ])
#
#         self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
#         self.relu = nn.ReLU()
#
#         self.use_residual = residual
#         if residual:
#             self.residual = nn.Sequential(*[
#                 Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
#                                   kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels),
#                 nn.ReLU()
#             ])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
#         org_x = x
#         if self.use_bottleneck:
#             x = self.bottleneck(x)
#         x = self.conv_layers(x)
#
#         if self.use_residual:
#             x = x + self.residual(org_x)
#         return x

def main():
    num_blocks = 3
    in_channels = 1
    out_channels = 16
    bottleneck_channels = 8
    kernel_sizes = 60
    use_residuals = True
    num_pred_classes = 1
    model = InceptionTime(num_blocks=num_blocks, in_channels=in_channels,
                           out_channels=out_channels, bottleneck_channels=bottleneck_channels,
                           kernel_sizes=kernel_sizes, use_residuals=use_residuals,
                           num_pred_classes=num_pred_classes)
    return model

class Inception(nn.Module):
    def __init__(self, in_channels, bottleneck=32, kernel_size=60, nb_filters=64):
        '''Inception
        '''
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, padding='same') if bottleneck and in_channels > 1 else lambda x: x
        d_mts = bottleneck or in_channels # dimensionality of multivariate time-series
        conv_layers = []
        kss = [kernel_size // (2**i) for i in range(3)]
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(d_mts, nb_filters, kernel_size=kss[i], padding='same')
                )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding='same')
        self.conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1)
        self.BN = nn.BatchNorm1d(nb_filters * 4)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0:
                out = out_
            else:
                out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 nb_filters=32, residual=True, depth=6):
        super().__init__()
        self.residual = residual
        self.depth = depth
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(depth):
            inc_mods.append(
                Inception(Cin if d == 0 else nb_filters * 4, bottleneck=bottleneck if d > 0 else 0,ks=ks,
                          nb_filters=nb_filters))
            if self.residual and d % 3 == 2:
                res_layers.append(shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4))
                res += 1
            else: res_layer = res_layers.append(None)
        self.inc_mods = nn.ModuleList(inc_mods)
        self.res_layers = nn.ModuleList(res_layers)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x



if __name__ == '__main__':
    main()
