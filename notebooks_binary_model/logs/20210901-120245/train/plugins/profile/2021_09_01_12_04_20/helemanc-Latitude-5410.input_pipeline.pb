	??R^+?@??R^+?@!??R^+?@	?̖??@?̖??@!?̖??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??R^+?@?hW!?'??AT5A?}?@Y?-c}??*	?Zd?c@2K
Iterator::Model::Map?C3O?)??!?2]?:N@) ?E
e???1Ns?1?[A@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2ճ ??q??!?~k?.u9@)ճ ??q??1?~k?.u9@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?f??I}??!?*???5@)????Đ??1?@???1@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap????Ǔ?!؛?W?(@)????Ì?1?̡(?!@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensork-?B;?y?!mw?S?@)k-?B;?y?1mw?S?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?4?($?u?!cfR???
@)?4?($?u?1cfR???
@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip9?#+???!9?7<d?B@)#?-?R\u?1?|l-??
@:Preprocessing2F
Iterator::Model???hW!??!?{?ÛJO@)5?+-#?n?1??F@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?̖??@I?I??PX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?hW!?'???hW!?'??!?hW!?'??      ??!       "      ??!       *      ??!       2	T5A?}?@T5A?}?@!T5A?}?@:      ??!       B      ??!       J	?-c}???-c}??!?-c}??R      ??!       Z	?-c}???-c}??!?-c}??b      ??!       JCPU_ONLYY?̖??@b q?I??PX@