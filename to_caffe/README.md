# onnx2caffe

- 安装我的[caffe_plus](https://github.com/jnulzl/caffe_plus) ，并编译pycaffe接口

这里主要是为了支持upsample layer

- 设置PYTHONPATH环境变量

 ```shell
 export PYTHONPATH=ROOT/caffe_plus/python:$PYTHONPATH # caffe_plus including upsample layer
 ```

- 切换到to_caffe/目录：

```shell
cd to_caffe/
```

- 执行：

```shell
python convertCaffe.py weights/ONNX_MODEL_FILE # 此处的onnx模型必须是三个输出形式的
```

# caffe2nnie

- prototxt修改

由于我自己实现的upsample层的参数名称与nnie支持的upsample层的不一致，需要进行如下更改

```shell
  ......
  upsample_param {
    height_scale: 2
    width_scale: 2
    mode: NEAREST
  }
  ......  
```

——>


```shell
  ......
  upsample_param {
	scale:2
  }
  ......  
```

- caffe2wk

用[nnie_mapper](https://github.com/jnulzl/nnie_mapper)进行caffe2wk
