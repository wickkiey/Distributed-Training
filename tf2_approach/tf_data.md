# Tensorflow 2x (tf.data)

## Why input pipeline

1. Data might not fit into memory
2. Data might require pre-processing
3. To use Hardware efficiently
4. Decouple data loading and pre-processing from distribution. 


## Parallelism 
To enhance the input pipeline performance we can define the `num_parallel_calls` argument. 

`interleave` to load the data faster
`map` to preprocess faster

By default, `tf.data.experimentat.AUTOTUNE` will take care according to the environment, but we can fine-tune manually also if needed. 


## tf.data Options

We can specify options to the tf.data. 

options = tf.data.Options()
options.experimenta_optimization.map_parallelization = True
dataset = dataset.with_options(options)  # Applying options to dataset


## Canned datasets. 

Tensorflow provides various datasets on multiple domains (text, image, audio, video). 
we can just load and exertiment the transformation. 

