# Tensorflow 2x Distributed Training

Reference : https://www.tensorflow.org/guide/distributed_training


### API

`tf.distribute.Strategy`
* Multiple GPU
* Multiple Machines
* Multiple TPUs

### Program Execution methods

* Eager execution (Not recommended)
* Tensorflow Funtions


### Types of Strategy

* Synchronous Training (With Data parallelism)
In sync training, all workers train over different slices of input data in sync, and aggregating gradients at each step.
Supported via all-reduce.

* Asynchronous training
In async training, all workers are independently training over the input data and updating variables asynchronously.
This is through parameter server architecture 

* Hardware
Scale your training onto
1. multiple GPUs on one machine
2. multiple machines in a network (with 0 or more GPUs each), 
3. Cloud TPUs


### List of strategies. 
* MirroredStrategy	
* TPUStrategy	
* MultiWorkerMirroredStrategy	
* CentralStorageStrategy	
* ParameterServerStrategy



### MirroredStrategy (One Machine multiple GPU)
`tf.distribute.MirroredStrategy` supports synchronous distributed training on multiple GPUs on one machine.

* Creates replica on each GPU
* Each variable in the model is mirrored across all the replicas
* Together, these variables form a single conceptual variable called MirroredVariable

* Efficient all-reduce algorithms are used to communicate the variable updates across the devices.
* Default all-reduce method `NVIDIA NCCL` 

    `mirrored_strategy = tf.distribute.MirroredStrategy()`

* Controlling limited GPUs
    `mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])`

#### Cross device communication overrides. 
    
`mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())`

Variations
tf.distribute.HierarchicalCopyAllReduce 
tf.distribute.ReductionToOneDevice
