	?O Ũ@?O Ũ@!?O Ũ@	?|???@?|???@!?|???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?O Ũ@?<Y????A??ut?@YOʤ?6 ??*	?rh??u@2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?m???!?tE???L@)I0??Z
??1?C????K@:Preprocessing2K
Iterator::Model::Map??.\s??!?+6*??@)??!9????1ː??)9@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatV?1?Ҝ?!QUS?? @)Yک??`??1u|?>@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?5|???!;?????@)?5|???1;?????@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?^ ??!"V???P@)؞Y??v?1-F?v?=??:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicet	4?t?!IF??%??)t	4?t?1IF??%??:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????q?!??@????)?????q?1??@????:Preprocessing2F
Iterator::ModelD???XP??!?S???f@@)?????k?15???4 ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?|???@I1?!n??W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?<Y?????<Y????!?<Y????      ??!       "      ??!       *      ??!       2	??ut?@??ut?@!??ut?@:      ??!       B      ??!       J	Oʤ?6 ??Oʤ?6 ??!Oʤ?6 ??R      ??!       Z	Oʤ?6 ??Oʤ?6 ??!Oʤ?6 ??b      ??!       JCPU_ONLYY?|???@b q1?!n??W@