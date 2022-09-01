import sys
import copy
from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):
    with open(modelName, 'rb') as f:
        caffemodel = caffe_pb2.NetParameter()
        caffemodel.ParseFromString(f.read())

    # 兼容新旧版本
    # LayerParameter 消息中的 blobs 保存着可训练的参数
    
    HAS_WEIGHTS={'BatchNorm':'batch_norm_param', 'Convolution':'convolution_param', 'InnerProduct':'innerproduct_param'}
    caffe_model_copy = copy.copy(caffemodel)
    for item in caffemodel.layers:
        item.ClearField('blobs')    
    # print(dir(caffemodel.layer))
    for item in caffemodel.layer: 
        # print("BBBBBBBBBBBB : ", dir(item))
        # print("BBBBBBBBBBBB : ", help(item.HasField))
        # print("BBBBBBBBBBBB : ", item.HasField('blobs'))
        if item.type not in HAS_WEIGHTS:
            print('Delete : ' + item.name) 
            caffe_model_copy.layer.remove(item)

    for item in caffe_model_copy.layer:
        item.ClearField(HAS_WEIGHTS[item.type])
        item.ClearField('phase')
    
   
    # print(dir(caffemodel))
    with open(deployName, 'w') as f:
        f.write(str(caffe_model_copy))

if __name__ == '__main__':

    if 2 != len(sys.argv):
        print("Usage:\n\t " + sys.argv[0] + ' caffemodel_path')    
        sys.exit(0)

    modelName = sys.argv[1]
    deployName = modelName + '.prototxt'
    toPrototxt(modelName, deployName)
