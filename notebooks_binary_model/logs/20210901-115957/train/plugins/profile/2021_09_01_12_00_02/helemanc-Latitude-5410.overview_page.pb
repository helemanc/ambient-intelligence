?	??5\??@??5\??@!??5\??@	9Ũ??@9Ũ??@!9Ũ??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??5\??@?nض(??A??Z
H?@Yw1?t????*	23333?b@2K
Iterator::Model::Map??ᱟź?!Υ?+??Q@)~?$A???1???&,?M@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???	?_??!E[?i?.@)?A_z?s??1/?GA??&@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?CP5z5??!?{??HI%@)?CP5z5??1?{??HI%@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?'?.????!M*]g?"@)?8?j?3??1????7@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???=??w?!]|?bi@)???=??w?1]|?bi@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?ݓ??Zs?!R|[?j	@)?ݓ??Zs?1R|[?j	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip5??-</??!??
?;@)????8r?1i`??@:Preprocessing2F
Iterator::Model6 B\9{??!;JZ}R@)J'L5?f?1S飋???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no99Ũ??@Iֹ??:X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?nض(???nض(??!?nض(??      ??!       "      ??!       *      ??!       2	??Z
H?@??Z
H?@!??Z
H?@:      ??!       B      ??!       J	w1?t????w1?t????!w1?t????R      ??!       Z	w1?t????w1?t????!w1?t????b      ??!       JCPU_ONLYY9Ũ??@b qֹ??:X@Y      Y@q??fJzi??"?
device?Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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