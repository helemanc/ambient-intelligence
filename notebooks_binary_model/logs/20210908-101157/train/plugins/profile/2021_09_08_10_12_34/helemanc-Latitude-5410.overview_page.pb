?	^*6?u?@^*6?u?@!^*6?u?@	sΘ??w@sΘ??w@!sΘ??w@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^*6?u?@Qٰ??(??A?1^?j@Y????????*	*\????s@2K
Iterator::Model::Mapn2????!'IS@)??????1?.?;R@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ڊ?e???!z?}Y?.@)?ڊ?e???1z?}Y?.@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?cϞˤ?!?W?A??)@)J%<?ן??1<E*C?@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapw??-u???!??dmm@)@?#H?ؑ?1N???%I@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV20EH?Ύ?!)p'?g<@)0EH?Ύ?1)p'?g<@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????v?!)???Ώ??)????v?1)???Ώ??:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?4?Op??!~x2??5@)E?4fr?1?b9\????:Preprocessing2F
Iterator::Model5A?} R??!??|?H?S@) ?t???k?1????Q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9tΘ??w@I?9?@X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Qٰ??(??Qٰ??(??!Qٰ??(??      ??!       "      ??!       *      ??!       2	?1^?j@?1^?j@!?1^?j@:      ??!       B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JCPU_ONLYYtΘ??w@b q?9?@X@Y      Y@q?+6???	@"?
device?Your program is NOT input-bound because only 3.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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