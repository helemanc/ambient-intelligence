	/?$??@/?$??@!/?$??@	?h/)??@?h/)??@!?h/)??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/?$??@?v?$$Ү?A???;?@Y??q?&??*	????K?e@2K
Iterator::Model::MapI??r?S??!!E????R@)??n???1.Td?P?O@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??Y.???!S??{?&@)??Y.???1S??{?&@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???c?3??!?Z+??)@)??4?䚒?1?Wܕ?`%@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?:s	ߋ?!d?M?= @)??y7??1J1<?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?k?,	Ps?!???d?0@)?k?,	Ps?1???d?0@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?֦????!"hY?>7@)?8?ߡ(p?1??.?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??K?l?!<??? @)??K?l?1<??? @:Preprocessing2F
Iterator::ModelKXc'???!??iV?:S@)???<j?1l5w&???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?h/)??@I???^kcX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?$$Ү??v?$$Ү?!?v?$$Ү?      ??!       "      ??!       *      ??!       2	???;?@???;?@!???;?@:      ??!       B      ??!       J	??q?&????q?&??!??q?&??R      ??!       Z	??q?&????q?&??!??q?&??b      ??!       JCPU_ONLYY?h/)??@b q???^kcX@