	M?O?t@M?O?t@!M?O?t@	??=VX?@??=VX?@!??=VX?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O?t@6t??Pn??AܷZ'.?@Y????????*	+?d@2K
Iterator::Model::Map2?Mc{-??!???A?dM@)?i???1??1????2F@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??@?9w??!h?|??7@)?3??X???1 ????<3@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2V-?????!??`pQ-@)V-?????1??`pQ-@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapV??{L??!?a?'W?(@)???ם???1P#5??!@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???l }?!??N_?@)???l }?1??N_?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceX?|[?Tw?!??$O?\@)X?|[?Tw?1??$O?\@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip|?ʄ_???!?x5IXfC@)????@gr?1?wv7s_@:Preprocessing2F
Iterator::Model?ri??+??!o?ʶ??N@)?x#??o?1Aw?NR@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??=VX?@I?N=?cX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6t??Pn??6t??Pn??!6t??Pn??      ??!       "      ??!       *      ??!       2	ܷZ'.?@ܷZ'.?@!ܷZ'.?@:      ??!       B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JCPU_ONLYY??=VX?@b q?N=?cX@