?	?\?@?\?@!?\?@	???5Q@???5Q@!???5Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?\?@s?`????AU?z??@Y?%?ؽ?*	??ʡm`@2K
Iterator::Model::Map?
?.ȶ?!_=V???P@)??W)??1??I.v?I@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?G??|??!??Ā??0@)?G??|??1??Ā??0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatF????!-`??'1@)?k&?ls??1?P???,@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?V*????!????#@)?unڌӀ?1z?k?u@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?=ϟ6?s?!yC?W[9@)?=ϟ6?s?1yC?W[9@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?٭e2??!N?	j?>@)<J%<??o?1???@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
-???m?!??BWј@)
-???m?1??BWј@:Preprocessing2F
Iterator::Modeln5???!,?}?Q@)XWj1xh?1???ĝ.@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???5Q@I??kSvUX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?`????s?`????!s?`????      ??!       "      ??!       *      ??!       2	U?z??@U?z??@!U?z??@:      ??!       B      ??!       J	?%?ؽ??%?ؽ?!?%?ؽ?R      ??!       Z	?%?ؽ??%?ؽ?!?%?ؽ?b      ??!       JCPU_ONLYY???5Q@b q??kSvUX@Y      Y@q?m??}Y2@"?	
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
Refer to the TF2 Profiler FAQb?18.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 