?	,G?@?@,G?@?@!,G?@?@	r?V?Ԙ@r?V?Ԙ@!r?V?Ԙ@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$,G?@?@Z??????A??Tl̋@Y?6??:r??*	rh??|?_@2K
Iterator::Model::Map?'??Q??!=??,IHO@)G?I?ѯ?1d??y3~H@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??P???!?"f??5@)??v???1???\?v2@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??nf????!f???V(+@)??nf????1f???V(+@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap}?;l"3??!d-????!@)?i?*?~?1H?uq?+@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zipj?? ?m??!??t??CA@)HP?s?r?1?<޳?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????g?r?!???K?@)????g?r?1???K?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice4???HLp?!?mA	@)4???HLp?1?mA	@:Preprocessing2F
Iterator::Model??kC??!)?Ž^P@)Lo.2n?1P1??D>@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9q?V?Ԙ@I\J?X9X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z??????Z??????!Z??????      ??!       "      ??!       *      ??!       2	??Tl̋@??Tl̋@!??Tl̋@:      ??!       B      ??!       J	?6??:r???6??:r??!?6??:r??R      ??!       Z	?6??:r???6??:r??!?6??:r??b      ??!       JCPU_ONLYYq?V?Ԙ@b q\J?X9X@Y      Y@q	x~Q??C@"?	
device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?39.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 