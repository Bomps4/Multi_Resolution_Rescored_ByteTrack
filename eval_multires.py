import argparse
import os
import random 
import torch

import warnings 
from loguru import logger
from Testing_library.utils.launch import launch
from Testing_library.utils.env import *
from parser_general_options import set_parser_option
from Testing_library.Experiments.exp_getter import get_exp
import datetime
from Testing_library.utils.logger import setup_logger



@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        #cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    if(args.devices*args.num_machines>1):
    # set environment variables for distributed training
        configure_nccl()
        configure_omp()
    
    trainer = exp.get_trainer(args,val=True)


if __name__ == "__main__":
    configure_module()
    parser = argparse.ArgumentParser("EVAL Parser")
    parser=set_parser_option(parser)
    
    args = parser.parse_args()

    exp = get_exp(args.exp_file, args.name)
    
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    #os.chdir('CNN_Training/')
    print()
    new_dir='Testing_library/Evaluator_outputs/'+str(exp.exp_name)+'_'+str(datetime.datetime.now())+'/'
    os.mkdir(new_dir)
    
    exp.saving_dir=new_dir
    setup_logger(
                new_dir,
                distributed_rank=0,
                filename=new_dir+"train_log.txt",
                mode="a",
            )

    

    

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    dist_url = "auto" if args.dist_url is None else args.dist_url 
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
