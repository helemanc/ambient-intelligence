?	t?3?.
@t?3?.
@!t?3?.
@	???|sA@???|sA@!???|sA@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$t?3?.
@?3?IbI??A?d?`T2@Y6??\??*	?Zd;f@2K
Iterator::Model::Map?t_?l??!/s6/+Q@)e??????1It?v@?I@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2>{.S????!(?n?+60@)>{.S????1(?n?+60@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatQg?!?{??!Er??X1@)??-???1?e?	?.@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapG 7???!Q?˄?$@)?&k?C4??1?|?k?o@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????w?!?cXD?	@)?????w?1?cXD?	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipWZF?=???!є*???=@)Y4???r?1???"@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??҈?}n?!???_? @)??҈?}n?1???_? @:Preprocessing2F
Iterator::Model??v?$$??!?Zu?B?Q@)?-?\ok?1O??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???|sA@IW?0???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?3?IbI???3?IbI??!?3?IbI??      ??!       "      ??!       *      ??!       2	?d?`T2@?d?`T2@!?d?`T2@:      ??!       B      ??!       J	6??\??6??\??!6??\??R      ??!       Z	6??\??6??\??!6??\??b      ??!       JCPU_ONLYY???|sA@b qW?0???W@Y      Y@q?????"@"?
both?Your program is POTENTIALLY input-bound because 3.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 