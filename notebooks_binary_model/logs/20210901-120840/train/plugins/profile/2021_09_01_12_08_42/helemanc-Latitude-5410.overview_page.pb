?	(__??@(__??@!(__??@	?Hϻ??@?Hϻ??@!?Hϻ??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$(__??@?A	3m???A|
???@Y6??D.8??*	?t??j@2K
Iterator::Model::Map~?N?Z???!i
^?,?E@)?ؗl<ز?1????HA@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?&????!"/?BD@)Ț?A?"??1g??? ?=@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Rb????!??܃;%@)?Rb????1??܃;%@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??ڦx\??!^h?"@)??ڦx\??1^h?"@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?@?v??!ŏ?o&@)??-c}??1oÀ?f?!@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip}?H?F???!sj??-;K@)*6?u?!{?1?????@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice£?#??s?!V
?s=@)£?#??s?1V
?s=@:Preprocessing2F
Iterator::Model.??M?Ҹ?!??P??F@)?U??6ol?1]d?W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?Hϻ??@I??!?:PX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?A	3m????A	3m???!?A	3m???      ??!       "      ??!       *      ??!       2	|
???@|
???@!|
???@:      ??!       B      ??!       J	6??D.8??6??D.8??!6??D.8??R      ??!       Z	6??D.8??6??D.8??!6??D.8??b      ??!       JCPU_ONLYY?Hϻ??@b q??!?:PX@Y      Y@q???-? @"?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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