?	-z?mC@-z?mC@!-z?mC@	?~u?@?~u?@!?~u?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-z?mC@!??nJ??AQ??ڦx@Y7ݲC????*	??Q?j@2K
Iterator::Model::Mapw.?????!%8??)O@)?:?????1?Q?&?	G@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?fc%?Y??!~?a?@@0@)?fc%?Y??1~?a?@@0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatD??{???!??)h??2@)??v????1?t??!-@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapqu ?]???!:?? S)@)nQf?L2??1??Џ?
!@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???????!r؃H{@)???????1r؃H{@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?2Wղ?!?B?W?A@)?c???_??1k=>,?E@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceH4?"??!??!? @)H4?"??1??!? @:Preprocessing2F
Iterator::ModelU?2?F??!?^T.P@)	4??yt?1????v-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?~u?@ITd!X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!??nJ??!??nJ??!!??nJ??      ??!       "      ??!       *      ??!       2	Q??ڦx@Q??ڦx@!Q??ڦx@:      ??!       B      ??!       J	7ݲC????7ݲC????!7ݲC????R      ??!       Z	7ݲC????7ݲC????!7ݲC????b      ??!       JCPU_ONLYY?~u?@b qTd!X@Y      Y@qLތU?X9@"?	
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?25.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 