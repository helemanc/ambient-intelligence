?	9
3f@9
3f@!9
3f@	T?O@T?O@!T?O@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$9
3f@??9y?	??A?CV?@YVa3????*	y??/?v@2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?;????!?$??I@)ΎT??E??1m?K?YI@:Preprocessing2K
Iterator::Model::Map?ԱJ陾?!?OO]?~@@)??\????1] ??v?8@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat4iSu?l??!c*??,(@)`???~???1wJ??tK%@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??T?????!e?ڍ?? @)??T?????1e?ڍ?? @:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?D??Ӝ??!h???z?P@):ZՒ?r??15?)q?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceo??\??v?!?0??T??)o??\??v?1?0??T??:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????`u?!a?????)????`u?1a?????:Preprocessing2F
Iterator::Model??A????!05?^
?@@)?1>?^?m?1??'@??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9T?O@I?_???W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??9y?	????9y?	??!??9y?	??      ??!       "      ??!       *      ??!       2	?CV?@?CV?@!?CV?@:      ??!       B      ??!       J	Va3????Va3????!Va3????R      ??!       Z	Va3????Va3????!Va3????b      ??!       JCPU_ONLYYT?O@b q?_???W@Y      Y@ql??*?/5@"?	
device?Your program is NOT input-bound because only 4.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?21.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 