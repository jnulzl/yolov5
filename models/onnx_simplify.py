import shutil
import os
import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')

import torch
import argparse
import onnx
import onnx.utils
from onnxsim import simplify



def optimize_onnx(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        # infile = infile.replace('.onnx', '.unoptimized.onnx')
        # shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    optimized_model = onnx.optimizer.optimize(model)
    onnx.save(optimized_model, outfile)


def check_onnx(modelfile):
    model = onnx.load(modelfile)
    onnx.checker.check_model(model)


def polish_onnx(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        # infile = infile.replace('.onnx', '.unpolished.onnx')
        # shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    polished_model = onnx.utils.polish_model(model)
    onnx.save(polished_model, outfile)


def simplify_onnx(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        # infile = infile.replace('.onnx', '.unsimplified.onnx')
        # shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, outfile)


def load_model(model_config, weights, device=torch.device('cpu')):
    # Load model
    model = Model(model_config).to(device)
    ckpt = torch.load(weights, map_location=device)
    ckpt['model'] = {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model


def remove_initializer_from_input(input_model, output=None):
    if output is None:
        output = input_model.replace('.onnx', '.remove_initializer_from_input.onnx')

    model = onnx.load(input_model)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output)
