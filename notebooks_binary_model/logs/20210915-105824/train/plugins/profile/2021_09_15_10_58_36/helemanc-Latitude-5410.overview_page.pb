?	+????@+????@!+????@	x?$l?|@x?$l?|@!x?$l?|@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+????@???Li???A	????@Y;?? ?>??*	+??·c@2K
Iterator::Model::Mapz8???n??!)??ګJM@)p]1#???1??J?c+F@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatMI???*??![?]??u:@)?|A??1?5?yGN5@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2q!??Fʖ?!6?sJ },@)q!??Fʖ?16?sJ },@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMappw?n?Ќ?!??Px?"@)?	?s3??1?[@@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Д?~??![ۺ?x?@)??Д?~??1[ۺ?x?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zipq??Ů?!m?t??;C@)?O?I?5s?1?dK?@:Preprocessing2F
Iterator::Model.?R????!?p?T,?N@)??̔??r?1?V;??@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?۠?[;q?!?,?)i?@)?۠?[;q?1?,?)i?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9x?$l?|@I?۞?lX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Li??????Li???!???Li???      ??!       "      ??!       *      ??!       2		????@	????@!	????@:      ??!       B      ??!       J	;?? ?>??;?? ?>??!;?? ?>??R      ??!       Z	;?? ?>??;?? ?>??!;?? ?>??b      ??!       JCPU_ONLYYx?$l?|@b q?۞?lX@Y      Y@q????d?.@"?	
device?Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?15.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 