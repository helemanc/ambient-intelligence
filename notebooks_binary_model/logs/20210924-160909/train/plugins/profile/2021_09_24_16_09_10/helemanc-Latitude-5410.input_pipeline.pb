	?P29??
@?P29??
@!?P29??
@	T??G??@T??G??@!T??G??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?P29??
@k??"ڶ?A??@?	@YۆQ<???*	MbX9a@2K
Iterator::Model::Map?_??ME??!sɓQ??O@)Ƣ??dp??1:?????G@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2????S??!r6Z?׻0@)????S??1r6Z?׻0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?Md????!?.??+?0@)f/?N[??1N*????+@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?g???c??!?&yvb*@)??O?s'??1?fZ??S!@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?Ǚ&l?y?!3?=?@)?Ǚ&l?y?13?=?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip~? ?}??!`?q6?@@)?I?pt?1?Â???@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???,??p?!X?XHK@)???,??p?1X?XHK@:Preprocessing2F
Iterator::Model?'????!P????P@)0??!?j?1ђ??c@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9T??G??@I???8!X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k??"ڶ?k??"ڶ?!k??"ڶ?      ??!       "      ??!       *      ??!       2	??@?	@??@?	@!??@?	@:      ??!       B      ??!       J	ۆQ<???ۆQ<???!ۆQ<???R      ??!       Z	ۆQ<???ۆQ<???!ۆQ<???b      ??!       JCPU_ONLYYT??G??@b q???8!X@