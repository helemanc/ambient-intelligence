	Z??լ?@Z??լ?@!Z??լ?@	دdk+{@دdk+{@!دdk+{@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z??լ?@0??乾??A????4?	@Y??kzPP??*	'1?*e@2K
Iterator::Model::MapK?b??¶?!?8??7AJ@)?P?f??15?c???B@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???!o??!U?I]?V=@)?????1?86s9@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2(?H0?̚?!j6j?;?.@)(?H0?̚?1j6j?;?.@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap???n???!Dm5???)@)m??~????1]<?y`0#@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?zNz??z?!?n?)	@)?zNz??z?1?n?)	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipP??@ֳ?!%?????F@)?PoF?w?1?bF??t@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??|??w?!????L2@)??|??w?1????L2@:Preprocessing2F
Iterator::Modelj?0
???!?W),
K@)pz???g?1f?c?R???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ׯdk+{@I?ڤ?&$X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0??乾??0??乾??!0??乾??      ??!       "      ??!       *      ??!       2	????4?	@????4?	@!????4?	@:      ??!       B      ??!       J	??kzPP????kzPP??!??kzPP??R      ??!       Z	??kzPP????kzPP??!??kzPP??b      ??!       JCPU_ONLYYׯdk+{@b q?ڤ?&$X@