	??%s,?@??%s,?@!??%s,?@	?/>?@?/>?@!?/>?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??%s,?@?Fx$??A??V??@YXuV?1??*	???Sg@2K
Iterator::Model::Map?F?@??!??z=R@){?\?&???1v-?zq?P@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?????H??!2????&@)?? ??F??1bRD"@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?}U.T???!7???@'@)	8?*5{??1?
MF?l!@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?9y?	???!?)<??:@)?9y?	???1?)<??:@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceaE|v?!?]?O@)aE|v?1?]?O@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?
(?ӧ?!?I1?L19@)?Ց#??q?1?cB?_?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~?k?,	p?!??m^? @)~?k?,	p?1??m^? @:Preprocessing2F
Iterator::Model"?? >???!??sǬ?R@)j?!?
l?1?_o9ӥ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?/>?@Im?l?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Fx$???Fx$??!?Fx$??      ??!       "      ??!       *      ??!       2	??V??@??V??@!??V??@:      ??!       B      ??!       J	XuV?1??XuV?1??!XuV?1??R      ??!       Z	XuV?1??XuV?1??!XuV?1??b      ??!       JCPU_ONLYY?/>?@b qm?l?W@