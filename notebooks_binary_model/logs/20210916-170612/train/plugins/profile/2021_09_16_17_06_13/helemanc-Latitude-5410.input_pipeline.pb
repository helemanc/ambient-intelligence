	34???@34???@!34???@	?j?E??@?j?E??@!?j?E??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$34???@>?ͨ???A
ܺ??*@Yb??h????*	?O??n?e@2K
Iterator::Model::MapU?-?????!?,n?K/P@)?}??ŉ??1?^0???J@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??t_Μ?!tCC?L0@)S#?3????1??Nqݗ(@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2k???@??!~??&??&@)k???@??1~??&??&@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?x?????!T>?P?G,@)'??@j??1???%@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!?> ?M|?!,o:!@)!?> ?M|?1,o:!@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???/fKv?!-?V;	@)???/fKv?1-?V;	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?4?BX???!?ed??@@)L???<u?1gDoX)	@:Preprocessing2F
Iterator::Model?w??Dg??!6?M	|?P@)?O:?`?i?1^?w??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?j?E??@I??hXX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>?ͨ???>?ͨ???!>?ͨ???      ??!       "      ??!       *      ??!       2	
ܺ??*@
ܺ??*@!
ܺ??*@:      ??!       B      ??!       J	b??h????b??h????!b??h????R      ??!       Z	b??h????b??h????!b??h????b      ??!       JCPU_ONLYY?j?E??@b q??hXX@