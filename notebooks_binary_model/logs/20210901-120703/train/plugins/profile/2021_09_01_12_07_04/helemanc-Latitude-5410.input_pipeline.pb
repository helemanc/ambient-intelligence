	E??Ӝ,@E??Ӝ,@!E??Ӝ,@	?֕???@?֕???@!?֕???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$E??Ӝ,@?)?D/???AA????@Y??m4????*	????xKr@2K
Iterator::Model::Map?? Z+???!p?1?~N@)K????1:=I@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatq?::?F??!s?x?34@)???????1ܢ??HY.@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?e?ikD??!??U?%@)?e?ikD??1??U?%@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?5Φ#???!?T
?*@)qh?.???1???n??"@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm???"??!\?*P?@)m???"??1\?*P?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(?hr1??!??^ld@)(?hr1??1??^ld@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??`??>??!???-B@)?}?֤ۂ?1????d*	@:Preprocessing2F
Iterator::Model?aMeQ???!??1?O@)?}?e???1?W??2@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?֕???@I???er?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)?D/????)?D/???!?)?D/???      ??!       "      ??!       *      ??!       2	A????@A????@!A????@:      ??!       B      ??!       J	??m4??????m4????!??m4????R      ??!       Z	??m4??????m4????!??m4????b      ??!       JCPU_ONLYY?֕???@b q???er?W@