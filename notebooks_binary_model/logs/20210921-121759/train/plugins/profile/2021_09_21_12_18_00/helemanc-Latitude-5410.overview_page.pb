?	??:??t@??:??t@!??:??t@	?I]J?Q???I]J?Q??!?I]J?Q??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??:??t@c??^'???A$??ŋ?@Y*???
ս?*	?S㥛c@2K
Iterator::Model::Map?Z??8???!V?????K@)p$?`S???1?c???F@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?7?Q????! ?[??;@){Cr??1??a?/?7@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?	0,??!6e???#@)?	0,??16e???#@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapiR
?????!?o3?Ȝ&@)?ó??1??!q?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????Đ|?!?ɶr N@)????Đ|?1?ɶr N@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor؃I??	y?!??瓕@)؃I??	y?1??瓕@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?s34???!׵1$OLE@)?2??A?v?1?5ow'@:Preprocessing2F
Iterator::Model0??e??!)J?۰?L@)ګ????e?1X?G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?I]J?Q??I؊????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c??^'???c??^'???!c??^'???      ??!       "      ??!       *      ??!       2	$??ŋ?@$??ŋ?@!$??ŋ?@:      ??!       B      ??!       J	*???
ս?*???
ս?!*???
ս?R      ??!       Z	*???
ս?*???
ս?!*???
ս?b      ??!       JCPU_ONLYY?I]J?Q??b q؊????X@Y      Y@q??W????"?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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