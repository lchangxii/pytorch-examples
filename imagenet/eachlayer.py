
import msgpack
import msgpack_numpy as m
m.patch()
import numpy as np
def save_msgp( tensor , target_dir, name ):
    if tensor == None:
        return
    numpy_array = tensor.cpu().detach().numpy()
    data = {
        "input" : numpy_array
        }
    with open(os.path.join(target_dir,"%s.msgp"%name),"wb") as bf:
        bf.write( msgpack.packb(data, use_bin_type=True) )
    shape = tensor.shape

import json

def save_config(layer,target_dir,name,inputs,output,layer_idx):
    data = dict()
    #kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    if hasattr(layer,"stride"):
        data["stride"] =  layer.stride
    if hasattr(layer,"padding"):
        data["padding"] = layer.padding
    if hasattr(layer,"kernel_size"):
        data["kernel_size"] = layer.kernel_size
    #print(dir(layer))
    if hasattr(layer,"in_channels"):
        data["in_channels"] = layer.in_channels
    if hasattr(layer,"out_channels"):
        data["out_channels"] = layer.out_channels
    data["layer_idx"] = layer_idx
    #if hasattr(layer,""):
    #print(layer)
    #exit(0)

    input_shape = []
    for input in inputs:
        if input is None:
            input_shape.append([0])
        else:
            input_shape.append(input.shape)
    data["inputs_shape"] = input_shape
    
    output_shape = []
    for output_elem in output:
        if output_elem is None:
            output_shape.append([0])
        else:
            output_shape.append(output_elem.shape)

    data["outputs_shape"] = output_shape
    #print(str(layer))
    #data["name"] = layer.name()
    ###
    with open(os.path.join(target_dir,"%s.json"%name),"w") as f:
        json.dump(data,f,indent=4)
#        f.write( str(shape ))



def get_activation(layer_idx,target_dir,name,if_exit):
    def hook(model, input2, output):
#        activation[name] = output.detach()
#        output.detach()
        #print(input2)
#        print (name,if_exit)
        flush_input = False
        if "conv" in name :
#            torch.save(model.weight, 'weight.pt')
#            torch.save(model.bias, 'bias.pt')
            save_msgp( model.bias,target_dir,"%s_bias"%name )
            save_msgp( model.weight,target_dir,"%s_weight"%name )
#            if "conv2d0_back" == name or name == "conv2d3_back":
            #    print (input2)
#            if "conv2d3" == name :
#                print (input2)
#            save_msgp( input2[0],target_dir,"%s_input"%name )
            flush_input = True
#            print(input2[0])
#        elif "relu" in name:
#                torch.save(output, 'tensor.pt')
            #save_msgp( output,target_dir,"%s_input"%name )
        elif "maxpool" in name :
#            save_msgp( input2[0],target_dir,"%s_input"%name )
            flush_input = True
        elif "avgpool" in name :
            #save_msgp( input2[0],target_dir,"%s_input"%name )
            flush_input = True
            print(dir(model))
        elif "dense" in name :
#            save_msgp( input2[0],target_dir,"%s_input"%name )
            save_msgp( model.bias,target_dir,"%s_bias"%name )
            save_msgp( model.weight,target_dir,"%s_weight"%name ) 
            flush_input = True
        if flush_input:
            for input_i, input_tmp in enumerate(input2):
                if input_tmp != None:
                    save_msgp( input_tmp ,target_dir,"%s_input_%d"%(name, input_i )) 
            for output_i, output_tmp in enumerate(output):
                if output_tmp != None:
                    save_msgp( output_tmp ,target_dir,"%s_output_%d"%(name, input_i )) 

            save_config(model,target_dir,name,input2,output,layer_idx)
            #print(output)
        if if_exit:
            #print (input2)
            print (output)
            numzero = 0
            flat_output = output[0].view(-1)
            for value in flat_output:
                elem = value.item()
                if elem == 0 :
                    numzero += 1
            print (numzero,len(flat_output))
            exit(0)
        
    return hook

import os
def collect(layers,iteration):
#    for #print(l)
    target_dir = "collect%s"%iteration
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    layer_num = 0    
    conv_layer_num = 0
    avgpool_layer_num = 0 
    maxpool_layer_num = 0
    dense_layer_num = 0
    if_exit = False
    for name,m in layers:
        module_type = str( type(m) )
        if "torch.nn.modules" not in module_type:
            continue
        if "torch.nn.modules.container.Sequential" in module_type:
            continue
#        if_exit = layer_num == len(l) - 1
        if_exit = layer_num == 0
        layer_name = name
        enable_register = False
        if "Conv2d" in module_type:
            conv_name = "conv2d%s"%conv_layer_num
            conv_layer_num = conv_layer_num + 1
            layer_name = conv_name
            enable_register = True
            print (module_type)
#        elif "ReLU" in module_type:
#            layer_name = "relu%s"%layer_num
        elif "MaxPool2d" in module_type:
            layer_name = "maxpool%s"%maxpool_layer_num
            maxpool_layer_num += 1
            enable_register = True
        elif "AvgPool2d" in module_type:
            layer_name = "avgpool%s"%avgpool_layer_num
            avgpool_layer_num += 1
            enable_register = True
        elif "Linear" in module_type:
            layer_name = "dense%s"%dense_layer_num
            dense_layer_num += 1
            enable_register = True
        if enable_register : 
            m.register_forward_hook(get_activation(layer_num,target_dir, layer_name,False))
            layer_name = layer_name + "_back"
#        print (if_exit)
#        m.register_backward_hook(get_activation(target_dir, layer_name,if_exit))
            m.register_full_backward_hook(get_activation(layer_num,target_dir,layer_name,if_exit))
        layer_num += 1
 #       print(module_type)
