#ifndef __onnx_graph_H__
#define __onnx_graph_H__

#define __PREFIX(x) onnx_graph ## x
// Include basic GAP builtins defined in the Autotiler
#include "at_api.h"

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE onnx_graph_L3_Flash;
extern AT_DEFAULTFLASH_EXT_ADDR_TYPE Low_size_onnx_graph_L3_Flash;


#endif