	)? ??@)? ??@!)? ??@	??O??+@??O??+@!??O??+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$)? ??@?4????A~t??g?@Yk??=]??*	'1??b@2K
Iterator::Model::Mapg??ͺ?!??UuQ@)?	L?u??1?,1J@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2\u?)ɚ?!9!)r1@)\u?)ɚ?19!)r1@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatj???'??!?H{0D).@)?k?F=D??1|?f??)@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapb?k_@/??!??cG["@)?:?f??1ؖ???1@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?W?\y?!???<??@)?W?\y?1???<??@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zipo.2??!T?;@)e????s?1?m}??	@:Preprocessing2F
Iterator::Model?=~oӻ?!???jzR@)dZ???Zp?1??Z??M@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?KU??o?!U?Q?5B@)?KU??o?1U?Q?5B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??O??+@I????FX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?4?????4????!?4????      ??!       "      ??!       *      ??!       2	~t??g?@~t??g?@!~t??g?@:      ??!       B      ??!       J	k??=]??k??=]??!k??=]??R      ??!       Z	k??=]??k??=]??!k??=]??b      ??!       JCPU_ONLYY??O??+@b q????FX@