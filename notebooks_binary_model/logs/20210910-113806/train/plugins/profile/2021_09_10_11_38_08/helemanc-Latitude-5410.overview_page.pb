?	?&???@?&???@!?&???@	???6@???6@!???6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?&???@?Ye?????Aж?u??@Yf???ٿ?*	??(\??b@2K
Iterator::Model::Map?'-\Va??!˦>u?N@)%?j????1?쯘0G@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat3?뤾,??!*ci??=3@)???ؗ?1M"???r/@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2o?ŏ1??!?DJEq?.@)o?ŏ1??1?DJEq?.@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapXSYvQ??!?V???*@)?????1.??Y??!@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice+i?7>{?!?ppI?@)+i?7>{?1?ppI?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?:???Ru?!??  @)?:???Ru?1??  @:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??&?E'??!????A@)??"[As?1&?5?e	@:Preprocessing2F
Iterator::Model??E?T??!=?~w?P@)D0.sn?1??۾@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???6@IR^?ߗ?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Ye??????Ye?????!?Ye?????      ??!       "      ??!       *      ??!       2	ж?u??@ж?u??@!ж?u??@:      ??!       B      ??!       J	f???ٿ?f???ٿ?!f???ٿ?R      ??!       Z	f???ٿ?f???ٿ?!f???ٿ?b      ??!       JCPU_ONLYY???6@b qR^?ߗ?W@Y      Y@qd멲?>@"?	
device?Your program is NOT input-bound because only 4.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?30.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 