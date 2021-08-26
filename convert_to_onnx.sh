echo python export.py --weights $1 --img-size $2 --include 'onnx' --opset 11 --simplify --export_three_output

python export.py --weights $1  --img-size $2 --include 'onnx' --opset 11 --simplify --export_three_output

echo python export.py --weights $1 --img-size $2 --include 'onnx' --opset 11 --simplify

python export.py --weights $1 --img-size $2 --include 'onnx' --opset 11 --simplify
