
import torch
from loguru import logger
class DataPrefetcher(object):
    """
    the main idea behind DataPrefetcher is to fetch data in a asyncronous manner before the actual computation
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def free(self):
        self.next_input=None   
        self.next_target=None
        torch.cuda.empty_cache()
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            
            self.next_input = self.next_input.cuda(non_blocking=True)
            
            self.next_target=self.next_target['annotations'].cuda(non_blocking=True)#{'annotations':,"width":[i["width"]for i in self.next_target],"height":[i["height"]for i in self.next_target],"image_id":[i["image_id"]for i in self.next_target]}
            
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
