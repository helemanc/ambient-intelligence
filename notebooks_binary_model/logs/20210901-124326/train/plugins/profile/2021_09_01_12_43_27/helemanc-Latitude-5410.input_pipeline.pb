	?Ƕ8{@?Ƕ8{@!?Ƕ8{@	??7?@??7?@!??7?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Ƕ8{@?~?T?¨?A_^?}t?@Yj???????*	??|?5?m@2K
Iterator::Model::Map??jGq???!&O???P@)^=?1X??1?`m???L@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?ɧǶ??!? ?䨡7@)R?8?ߡ??16???E4@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?!o?????!?????%%@)?!o?????1?????%%@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapZI+?????!mz?6Y@)Pp??Ӏ?1Z?׷S?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?
E??S??!$?`ݟ?
@)?
E??S??1$?`ݟ?
@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?GR??в?!?ܨ\Z?>@)??~j?ts?1^?? @:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??66;r?!?j?0??)??66;r?1?j?0??:Preprocessing2F
Iterator::Model0??&???!???h?AQ@)?d??7ij?1Mj??s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??7?@I?I`Gn!X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~?T?¨??~?T?¨?!?~?T?¨?      ??!       "      ??!       *      ??!       2	_^?}t?@_^?}t?@!_^?}t?@:      ??!       B      ??!       J	j???????j???????!j???????R      ??!       Z	j???????j???????!j???????b      ??!       JCPU_ONLYY??7?@b q?I`Gn!X@