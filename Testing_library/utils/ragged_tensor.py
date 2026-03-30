import torch
from typing import List,Union
from loguru import logger
from .general_functions import all_equal

class RaggedTensor(object):
    """
    The current implementation expects a list of tensor that differ in at most one dimension
    (and so cannot be stacked toghether but can be concateneted along that dimension)
    Usecase are boxes or labels  
    """

    def __init__(self,tensors:Union[List[torch.Tensor],torch.Tensor],sizes:List[int]):
        super().__init__()
        if isinstance(tensors,torch.Tensor):
            self.tensor=tensors.clone()
        else:
            self.tensor=torch.cat(tensors,dim=0)
        
        self.sizes=sizes

    def __truediv__(self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                tensor = torch.cat([tensor/a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                tensor = self.tensor/a
        elif(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            tensor = torch.cat([tensor/a[idx] for idx,tensor in enumerate(spread)],dim=0)

        elif(isinstance(a,RaggedTensor)):
            assert len(self.sizes)==len(a.sizes) and all([i==j for i,j in zip(self.sizes,a.sizes)]),f'expecting the two Ragged tensor to have both the same total size and same distribution but got {self.sizes} and {a.sizes}'
            tensor=self.tensor / a.tensor

        else:
            tensor=self.tensor /a
                
        return RaggedTensor(tensor,self.sizes)

    def __mul__(self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                tensor = torch.cat([tensor*a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                tensor = self.tensor*a 
        elif(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            tensor = torch.cat([tensor*a[idx] for idx,tensor in enumerate(spread)],dim=0)

        elif(isinstance(a,RaggedTensor)):
            assert len(self.sizes)==len(a.sizes) and all([i==j for i,j in zip(self.sizes,a.sizes)]),f'expecting the two Ragged tensor to have both the same total size and same distribution but got {self.sizes} and {a.sizes}'
            tensor = self.tensor*a.tensor
        else:
            tensor=self.tensor *a
        
        return RaggedTensor(tensor,self.sizes)
    
    def __rmul__(self, a):
        return self.__mul__(a)

    def __sub__(self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                tensor = torch.cat([tensor-a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                tensor = torch.cat([tensor-a for idx,tensor in enumerate(spread)],dim=0)

        elif(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            tensor = torch.cat([tensor-a[idx] for idx,tensor in enumerate(spread)],dim=0)

        elif(isinstance(a,RaggedTensor)):
            assert len(self.sizes)==len(a.sizes) and all([i==j for i,j in zip(self.sizes,a.sizes)]),f'expecting the two Ragged tensor to have both the same total size and same distribution but got {self.sizes} and {a.sizes}'
            tensor =self.tensor - a.tensor
        
        else:
            tensor=self.tensor -a
    
        return RaggedTensor(tensor,self.sizes)

    def __add__(self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                tensor = torch.cat([tensor+a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                tensor = torch.cat([tensor+a for idx,tensor in enumerate(spread)],dim=0)

        elif(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            tensor = torch.cat([tensor+a[idx] for idx,tensor in enumerate(spread)],dim=0)
        
        elif(isinstance(a,RaggedTensor)):
            assert len(self.sizes)==len(a.sizes) and all([i==j for i,j in zip(self.sizes,a.sizes)]),f'expecting the two Ragged tensor to have both the same total size and same distribution but got {self.sizes} and {a.sizes}'
            tensor = self.tensor + a.tensor
        
        else:
            tensor = self.tensor + a
    
        return RaggedTensor(tensor,self.sizes)

    def __eq__(self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                return torch.cat([tensor==a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                return torch.cat([tensor==a for idx,tensor in enumerate(spread)],dim=0)

        if(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            return torch.cat([tensor==a[idx] for idx,tensor in enumerate(spread)],dim=0)

    def __lt__ (self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(a)}'
                return torch.cat([tensor<a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                return torch.cat([tensor<a for idx,tensor in enumerate(spread)],dim=0)

        if(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            return torch.cat([tensor<a[idx] for idx,tensor in enumerate(spread)],dim=0)

    def __gt__ (self,a):
        spread=self.tensor.split(self.sizes,dim=0)
        if(isinstance(a,torch.Tensor)):
            if(a.dim()>=2 or (a.dim()!=0 and a.shape[0]>1)):
                assert (a.shape[0]==len(self.sizes)),f'number of elements (batch size) in a ragged tensor and n of elements of operand must be the same got{a.shape[0]} and {len(self.sizes)}'
                return torch.cat([self.tensor>a[idx] for idx,tensor in enumerate(spread)],dim=0)
            else:
                return torch.cat([self.tensor>a for idx,tensor in enumerate(spread)],dim=0)
                
            
        if(isinstance(a,list)):
            assert len(self.sizes)==len(a),f'number of elements in a ragged tensor and n of elements of operand must be the same got{len(self.sizes)} and {len(a)}'
            return torch.cat([self.tensor>a[idx] for idx,tensor in enumerate(spread)],dim=0)
    
    def __getitem__(self,i):
        return self.tensor[i]
    
    def get_org_tensors(self):
        spread=self.tensor.split(self.sizes,dim=0)
        return spread

    def to(self,*args,**kwargs):
        self.tensor=self.tensor.to(*args,**kwargs)
        return self
    
    def cuda(self,device=None, non_blocking=False, memory_format=torch.preserve_format):
        self.tensor=self.tensor.cuda(device=device,non_blocking=non_blocking,memory_format=memory_format)
        return self
    
    def get_tensor(self):
        return self.tensor
    
    def __repr__(self):
        return f'RaggedTensor(tensor:{self.tensor.__repr__()},sizes:{self.sizes})'

    def unbind(self,dim):
        a=self.tensor.unbind(dim)
        return tuple((RaggedTensor(i,self.sizes) for i in a))
    
    def record_stream(self,stream):
        self.tensor.record_stream(stream)
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in {torch.stack}:
            assert all_equal([element.sizes for arg in args for element in arg]),'all ragged tensors must have the same number of splits and the splits must be equal'

            new_tensor=torch.stack([element.tensor for arg in args for element in arg],dim=kwargs['dim'])
            return RaggedTensor(new_tensor,args[0][0].sizes)

    




if __name__ == '__main__':
    a=torch.tensor([1,2,3,4,5,6,7])
    b=torch.tensor([8,9,10])
    new_divide=3
    ragged=RaggedTensor([a,b],[7,3])

   
    my_solution=ragged/new_divide
    print(my_solution)

    print(logger._core.handlers)
