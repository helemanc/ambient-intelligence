	?t{I?@?t{I?@!?t{I?@	a??
R
@a??
R
@!a??
R
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?t{I?@??5\䞲?A?4E?ӻ@Ym7?7M???*	9??v?Of@2K
Iterator::Model::Map?Fw;S??!"?T?Q@)?j?3??1gn켓?M@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??Դ?i??!?V߲;?(@)??Դ?i??1?V߲;?(@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat0?'???!s??e??(@)^???????1??:*?$@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap???]g??!??	?fk'@)ޫV&?R??1?m?yU#!@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice ??*Q?v?!???5E 	@) ??*Q?v?1???5E 	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??mr??!?B2	3?:@)??Os?"s?1????@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??IӠhn?!?^s?? @)??IӠhn?1?^s?? @:Preprocessing2F
Iterator::Model?*8???!Eo?=?OR@)#??fF?j?1?LS8z???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9a??
R
@I?dڪo-X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??5\䞲???5\䞲?!??5\䞲?      ??!       "      ??!       *      ??!       2	?4E?ӻ@?4E?ӻ@!?4E?ӻ@:      ??!       B      ??!       J	m7?7M???m7?7M???!m7?7M???R      ??!       Z	m7?7M???m7?7M???!m7?7M???b      ??!       JCPU_ONLYYa??
R
@b q?dڪo-X@