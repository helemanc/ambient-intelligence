	D?????@D?????@!D?????@	_??>?)@_??>?)@!_??>?)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$D?????@d?&????A??>?@Y?????*	?ʡE??`@2K
Iterator::Model::Map?bg
???!?c?'? Q@)??????1Bz??W?I@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?OU??X??!(??2M0@)?OU??X??1(??2M0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??? ????!?????K0@)˟o????1Q?T?;>+@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapT??~m??!N???0%@)xe?????1L?@???@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?n???q?!???m?	@)?n???q?1???m?	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipJ}Yک???!&???&?=@)R<??kp?1???.-?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorRH2?w?m?!)???f@)RH2?w?m?1)???f@:Preprocessing2F
Iterator::Model?????Z??!6[N??Q@)?X5s?g?1b?>??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9_??>?)@IzULd?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d?&????d?&????!d?&????      ??!       "      ??!       *      ??!       2	??>?@??>?@!??>?@:      ??!       B      ??!       J	??????????!?????R      ??!       Z	??????????!?????b      ??!       JCPU_ONLYY_??>?)@b qzULd?W@