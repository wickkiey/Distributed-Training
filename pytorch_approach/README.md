# PyTORCH Distributed Training

(Refer : https://pytorch.org/tutorials/beginner/dist_overview.html)


### Distributed Data-Parallel Training (DDP) 
is a widely adopted single-program multiple-data training paradigm. With DDP, the model is replicated on every process, and every model replica will be fed with a different set of input data samples. DDP takes care of gradient communications to keep model replicas synchronized and overlaps it with the gradient computations to speed up training.


### RPC-Based Distributed Training (RPC)
 is developed to support general training structures that cannot fit into data-parallel training, such as distributed pipeline parallelism, parameter server paradigm, and combination of DDP with other training paradigms. It helps manage remote object lifetime and extend autograd engine to beyond machine boundaries.

 ### Collective Communication (c10d) 
 library support sending tensors across processes within a group. It offers both collective communication APIs (e.g., all_reduce and all_gather) and P2P communication APIs (e.g., send and isend). DDP and RPC (ProcessGroup Backend) are built on c10d as of v1.6.0, where the former uses collective communications and the latter uses P2P communications. 


 ## Ideal scenario in real-time use-case 

Below incremental approach will be followed, 
1. Use single-device training, if the data and model can fit in one GPU, and the training speed is not a concern.
2. Use single-machine multi-GPU DataParallel, if there are multiple GPUs on the server, and you would like to speed up training with the minimum code change.
3. Use single-machine multi-GPU DistributedDataParallel, if you would like to further speed up training and are willing to write a little more code to set it up.
4. Use multi-machine DistributedDataParallel and the launching script, if the application needs to scale across machine boundaries.
5. Use torchelastic to launch distributed training, if errors (e.g., OOM) are expected or if the resources can join and leave dynamically during the training.



