	R?T	@R?T	@!R?T	@	{???@{???@!{???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$R?T	@ ?߽?ƴ?A??}???@Y??J
,??*	6^?IJc@2K
Iterator::Model::Map???VC???!]?@aQ@)????~??1Λ*??I@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??|?͍??!?u?(?+0@)??|?͍??1?u?(?+0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??P???!?Cꏽ/@)?XP?i??1?~c??)@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapOt	???!d?O?Q#@)g??/??1?"?oH@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip/?????!֓
??=@)fI??Z?v?1 ???@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??)??v?!RKW?յ@)??)??v?1RKW?յ@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[?O?r?!?B??@)?[?O?r?1?B??@:Preprocessing2F
Iterator::Model??r????!
[?X?Q@)??????q?1????Ǘ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9{???@I?o?C?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?߽?ƴ? ?߽?ƴ?! ?߽?ƴ?      ??!       "      ??!       *      ??!       2	??}???@??}???@!??}???@:      ??!       B      ??!       J	??J
,????J
,??!??J
,??R      ??!       Z	??J
,????J
,??!??J
,??b      ??!       JCPU_ONLYY{???@b q?o?C?W@