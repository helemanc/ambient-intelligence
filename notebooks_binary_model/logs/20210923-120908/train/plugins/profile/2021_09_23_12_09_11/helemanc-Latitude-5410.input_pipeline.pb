	???v?:@???v?:@!???v?:@	)*D??#@)*D??#@!)*D??#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???v?:@f-????AH¾?Dt@Yt???1??*	??C???@2K
Iterator::Model::Map_$??\
??!5?E?KW@)?????w??1? ??V@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?mQf?L??!?	?d??@)?mQf?L??1?	?d??@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?3??k???!P?k?o@)????]M??1"3?@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMappD??k???!?;!y?@)???_w???1j?y??:Preprocessing2F
Iterator::Modeld> Й4??!W2%W??W@)G6uu?1b????D??:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice⬈???q?!??y?!.??)⬈???q?1??y?!.??:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?A?p?-n?!????{??)?A?p?-n?1????{??:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip*p?܁??!?ڬ?j?@)]S ???m?1??2WO??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9)*D??#@I?z7?<?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f-????f-????!f-????      ??!       "      ??!       *      ??!       2	H¾?Dt@H¾?Dt@!H¾?Dt@:      ??!       B      ??!       J	t???1??t???1??!t???1??R      ??!       Z	t???1??t???1??!t???1??b      ??!       JCPU_ONLYY)*D??#@b q?z7?<?V@