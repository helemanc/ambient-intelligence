	c?D(?@c?D(?@!c?D(?@	6YqpK'@6YqpK'@!6YqpK'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$c?D(?@?S? Pű?A?2ı..@Yl_@/ܹ??*	?G?z*`@2K
Iterator::Model::MapҦ??\??!3@???!P@)?iO?9???1U???G@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??????!?V?I??0@)??????1?V?I??0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???2??!??Z?T?1@)㈵? ??1?q???,@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap??<??-??!??Vo(@)r?#D??1{?Ó? @:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceo??}U.t?!*?7	{@)o??}U.t?1*?7	{@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip???????!?q??F?@@)????8r?1)tO?m?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??/?p?!l??V?X	@)??/?p?1l??V?X	@:Preprocessing2F
Iterator::Model???[??!/?3?ܴP@)D???XPh?1w߰u?\@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no96YqpK'@I6u|??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S? Pű??S? Pű?!?S? Pű?      ??!       "      ??!       *      ??!       2	?2ı..@?2ı..@!?2ı..@:      ??!       B      ??!       J	l_@/ܹ??l_@/ܹ??!l_@/ܹ??R      ??!       Z	l_@/ܹ??l_@/ܹ??!l_@/ܹ??b      ??!       JCPU_ONLYY6YqpK'@b q6u|??X@