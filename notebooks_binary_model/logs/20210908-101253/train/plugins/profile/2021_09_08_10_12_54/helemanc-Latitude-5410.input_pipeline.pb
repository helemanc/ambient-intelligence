	N????{ @N????{ @!N????{ @	$?HF?@$?HF?@!$?HF?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$N????{ @?f׽??AJ(}!??@Y?=?U???*	@5^?Idm@2K
Iterator::Model::MapΊ??>??!?,??4S@)C?? ????1?9???jP@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2X<?H?ۚ?!?I^eO&@)X<?H?ۚ?1?I^eO&@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??,??2??!?p]?o@(@)E????ؘ?1?y?֨?$@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap܁:?э??!UӞ?;?@)?;?2Tń?1?P???@@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceޏ?/??x?!??.?~@)ޏ?/??x?1??.?~@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip>?h??!Zp?K?5@)7???ZDt?1|??? @:Preprocessing2F
Iterator::Model0??乾??!???@?S@)4,F]k?s?1? ? @:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??aMeq?!?K6???)??aMeq?1?K6???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9$?HF?@I????mXX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f׽???f׽??!?f׽??      ??!       "      ??!       *      ??!       2	J(}!??@J(}!??@!J(}!??@:      ??!       B      ??!       J	?=?U????=?U???!?=?U???R      ??!       Z	?=?U????=?U???!?=?U???b      ??!       JCPU_ONLYY$?HF?@b q????mXX@