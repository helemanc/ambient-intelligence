	??y??w@??y??w@!??y??w@	?C???@?C???@!?C???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??y??w@\???4??A?4?Ry?@Y8i???*??????i@)       =2K
Iterator::Model::Map??4}v??!e?S?#?J@)6?:???1?????E@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatK??z2???!է<?@@)?0
?Ƿ??1?V???>@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2???9??!?k[?F%@)???9??1?k[?F%@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap????????!]]Я#?"@)?W?\T??1W?=N?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip\?nK䂷?!????r@F@)?
ҌE?y?1??sq@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice#?Ƥ?w?!?Z?C?y@)#?Ƥ?w?1?Z?C?y@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?m??4r?!m???9;@)?m??4r?1m???9;@:Preprocessing2F
Iterator::ModelMM?7?Q??!8H
??K@)?iT?dk?1~??/???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?C???@IX??b+XX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	\???4??\???4??!\???4??      ??!       "      ??!       *      ??!       2	?4?Ry?@?4?Ry?@!?4?Ry?@:      ??!       B      ??!       J	8i???8i???!8i???R      ??!       Z	8i???8i???!8i???b      ??!       JCPU_ONLYY?C???@b qX??b+XX@