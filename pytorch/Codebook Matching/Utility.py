from pathlib import Path
import os.path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Library.Utility import *


def CreateMatrix(values):
    matrix = torch.zeros(values.shape[0], 4, 4, device=values.device)

    x = values[:, 3]
    y = values[:, 4]
    z = values[:, 5]
    tensor_0 = torch.zeros_like(y)
    tensor_1 = torch.ones_like(y)
    rotation_x = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, torch.cos(x), -torch.sin(x)]),
        torch.stack([tensor_0, torch.sin(x), torch.cos(x)]),
    ]).permute(2, 0, 1)
    rotation_y = torch.stack([
        torch.stack([torch.cos(y), tensor_0, torch.sin(y)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-torch.sin(y), tensor_0, torch.cos(y)]),
    ]).permute(2, 0, 1)
    rotation_z = torch.stack([
        torch.stack([torch.cos(z), -torch.sin(z), tensor_0]),
        torch.stack([torch.sin(z), torch.cos(z), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1]),
    ]).permute(2, 0, 1)
    rotation = torch.bmm(rotation_y, torch.bmm(rotation_x, rotation_z))

    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = values[:, :3]
    matrix[:, 3, 3] = tensor_1
    return matrix


def GetPositions(matrices):
    return matrices[:, :, :3, 3]


def GetForwards(matrices):
    return matrices[:, :, 2, :3]


def GetUpwards(matrices):
    return matrices[:, :, 1, :3]


def GetRights(matrices):
    return matrices[:, :, 0, :3]


def PositionsFrom(positions, transforms):
    return torch.add(transforms[:, :3, 3].unsqueeze(2), torch.matmul(transforms[:, :3, :3], positions.unsqueeze(2))).squeeze(-1)


def PositionsTo(positions, transforms):
    inverse = torch.inverse(transforms)
    return torch.add(inverse[:, :3, 3].unsqueeze(2), torch.matmul(inverse[:, :3, :3], positions.unsqueeze(2))).squeeze(-1)


def TransformationsTo(matrices, transforms):
    return torch.bmm(torch.inverse(transforms), matrices)


def TransformationsFrom(matrices, transforms):
    return torch.bmm(transforms, matrices)


class FKLayer(torch.nn.Module):
    def __init__(self, hierarchy):
        super().__init__()
        self.hierarchy = hierarchy
        self.pairs = []
        self.offsets = []
        for item in self.hierarchy:
            self.pairs.append([int(item[0]), int(item[1])])
            self.offsets.append(torch.tensor([float(item[2]), float(item[3]), float(item[4])]))

    def forward(self, params):
        transformations = []
        for index in range(len(self.hierarchy)):
            transformations.append(CreateMatrix(params[:, index, :]))
            if index > 0:
                transformations[-1][:, 3, 0] = self.offsets[index][0]
                transformations[-1][:, 3, 1] = self.offsets[index][1]
                transformations[-1][:, 3, 2] = self.offsets[index][2]

        for parent, child in self.pairs:
            if parent != -1:
                transformations[child] = torch.matmul(transformations[parent], transformations[child])

        return torch.stack(transformations, 1)


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y * self.alpha + self.beta


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y * self.alpha + self.beta


class LN_v3(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        return (x - mean) / std


class LN_v4(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y * self.alpha + self.beta


class PlottingWindow:
    def __init__(self, title, ax=None, min=None, max=None, cumulativeHorizon=100, drawInterval=100, yScale='linear'):
        plt.ion()
        warnings.filterwarnings("ignore", message="Attempt to set non-positive ylim on a log-scaled axis will be ignored.")
        _, self.ax = plt.subplots() if ax is None else ax
        self.Title = title
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.YMin = min
        self.YMax = max
        self.YRange = [sys.float_info.max if min is None else min, sys.float_info.min if max is None else max]
        self.Functions = {}
        self.Counter = 0
        self.YScale = yScale

    def Add(self, *args):
        for value, label in args:
            if label not in self.Functions:
                self.Functions[label] = ([], [])
            function = self.Functions[label]
            function[0].append(value)
            cumulative = sum(function[0][-self.CumulativeHorizon:]) / len(function[0][-self.CumulativeHorizon:])
            function[1].append(cumulative)
            if self.YMin is None:
                self.YRange[0] = min(self.YRange[0], 0.5 * cumulative)
            if self.YMax is None:
                self.YRange[1] = max(self.YRange[1], 2 * cumulative)

        self.Counter += 1
        if self.Counter >= self.DrawInterval:
            self.Counter = 0
            self.Draw()

    def Draw(self):
        self.ax.cla()
        self.ax.set_title(self.Title)
        for label, function in self.Functions.items():
            step = max(int(len(function[0]) / self.DrawInterval), 1)
            self.ax.plot(function[0][::step], label=label + " (" + str(round(self.CumulativeValue(label), 3)) + ")")
            self.ax.plot(function[1][::step], c=(0, 0, 0))
        self.ax.set_yscale(self.YScale)
        self.ax.set_ylim((self.YRange[0], self.YRange[1]))
        self.ax.legend()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1e-5)

    def Value(self, label=None):
        if label is None:
            return sum(x[0][-1] for x in self.Functions.values())
        return self.Functions[label][0][-1]

    def CumulativeValue(self, label=None):
        if label is None:
            return sum(x[1][-1] for x in self.Functions.values())
        return self.Functions[label][1][-1]

    def Print(self, digits=5):
        output = ""
        for name in self.Functions.keys():
            output += name + ": " + str(round(self.CumulativeValue(name), digits)) + " "
        print(output)


class RunningStats:
    def __init__(self, dims):
        self.dims = dims
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros(self.dims)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else np.ones(self.dims)

    def sigma(self):
        return np.sqrt(self.variance())


def GetFileID(file):
    return os.path.basename(os.path.dirname(file)) + "_" + os.path.basename(file)


def SaveONNX(path, model, input_size, input_names, output_names, dynamic_axes=None):
    FromDevice(model)
    torch.onnx.export(
        model,
        input_size,
        path,
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,
        opset_version=12,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    ToDevice(model)


def ToPyTorch(x):
    return ToDevice(torch.tensor(x, requires_grad=True))


def LoadTxtRaw(path, debug=False, lineCount=None, axis=None):
    print("Loading " + path)
    data = []
    with open(path) as file:
        pivot = 0
        for line in file:
            pivot += 1
            if debug:
                PrintProgress(pivot, lineCount)
            entry = line.rstrip().split(' ')
            if axis is not None:
                entry = entry[axis]
            data.append(entry)
    return data


def NormalizeBN(x, norm):
    if norm.weight is not None and norm.bias is not None:
        return (x - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps) * norm.weight + norm.bias
    return (x - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps)


def RenormalizeBN(x, norm):
    if norm.weight is not None and norm.bias is not None:
        return (x - norm.bias) / norm.weight * torch.sqrt(norm.running_var + norm.eps) + norm.running_mean
    return x * torch.sqrt(norm.running_var + norm.eps) + norm.running_mean


def PrintParameters(model, learnable=None):
    for name, param in model.named_parameters():
        if learnable is None:
            print(name, param)
        if learnable is True and param.requires_grad:
            print(name, param)
        if learnable is False and not param.requires_grad:
            print(name, param)


def FreezeParameters(model, value, names=None):
    for name, param in model.named_parameters():
        if names is None or name in names:
            param.requires_grad = not value


def GetParameters(model, learnable=None):
    params = []
    for name, param in model.named_parameters():
        if learnable is None:
            params.append((name, param))
        if learnable is True and param.requires_grad:
            params.append((name, param))
        if learnable is False and not param.requires_grad:
            params.append((name, param))
    return params


def GetLabelIndicesExclude(file, names):
    indices = []
    with open(file, "r") as handle:
        for index, value in enumerate(handle):
            valid = True
            if names is not None:
                for name in names:
                    if name in value:
                        valid = False
            if valid:
                indices.append(index)
    return torch.tensor(indices)


def GetLabelIndicesContain(file, names):
    indices = []
    with open(file, "r") as handle:
        for index, value in enumerate(handle):
            valid = False
            if names is not None:
                for name in names:
                    if name in value:
                        valid = True
            if valid:
                indices.append(index)
    return torch.tensor(indices)
