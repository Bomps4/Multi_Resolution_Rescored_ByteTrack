
import torch 

from ..Models.TransVOD_Lite_2.util.misc_multi import NestedTensor
# from ..Models.TransVOD_plusplus.util.misc import NestedTensor




def make_nested_tensor(inp):
    mask=originals['mask']
    # print(f'sono inp {type(inp)}')
    my_inp=NestedTensor(inp.unsqueeze(0),mask)
    inp=my_inp