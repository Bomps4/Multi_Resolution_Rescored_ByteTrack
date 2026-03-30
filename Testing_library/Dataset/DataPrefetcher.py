
import torch
from loguru import logger

def to_cuda_recursive(value):   
    if(isinstance(value,dict)):
        return {key:to_cuda_recursive(value[key]) for key in value}     
    else:
        return value.cuda(non_blocking=True)

def recursive_record_stream(value,stream):
    if(isinstance(value,dict)):
        return {key:recursive_record_stream(value[key],stream) for key in value}     
    else:
        value.record_stream(stream)
        return value

class DataPrefetcher(object):
    """
    the main idea behind DataPrefetcher is to fetch data in a asyncronous manner before the actual computation
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.len=len(loader)
        self.preload()
        
    def __len__(self):
        return self.len
    
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
            
            self.next_input=to_cuda_recursive(self.next_input)
            self.next_target=to_cuda_recursive(self.next_target)
            
            # self.next_target=self.next_target['annotations'].cuda(non_blocking=True)#{'annotations':,"width":[i["width"]for i in self.next_target],"height":[i["height"]for i in self.next_target],"image_id":[i["image_id"]for i in self.next_target]}
            
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
            # self.next_input = self.next_input.float()
            
    def __next__(self):
        if(self.next_input is None):
            raise StopIteration
        current_stream=torch.cuda.current_stream()
        current_stream.wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        
        

        if input is not None:
            input=recursive_record_stream(input,current_stream)
        if target is not None:
            target=recursive_record_stream(target,current_stream)

        
        self.preload()
        return input, target

    def __iter__(self):
        return self
    


if __name__ == '__main__':
    print(torch.cuda.is_available())
    cuda_stream=torch.cuda.Stream()
    a={'1':{'1.1':torch.ones(3,3),'1.2':torch.ones(4,4)},'2':{'2.1':torch.ones(3,3),'2.2':torch.ones(4,4)},'3':{'3.1':torch.ones(3,3),'3.2':torch.ones(4,4)}}
    print(a)
    
    a=to_cuda_recursive(a)
    torch.cuda.current_stream().wait_stream(cuda_stream)
    print([[a[i][b].device for b in a[i]] for i in a])
    print(a)