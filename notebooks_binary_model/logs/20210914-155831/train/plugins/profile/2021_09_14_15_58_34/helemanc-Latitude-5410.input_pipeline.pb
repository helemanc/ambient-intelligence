	??L?@??L?@!??L?@	MyF?@MyF?@!MyF?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??L?@КiQ??A;R}?E@Y?M*k??*	??K7?Me@2K
Iterator::Model::Map\?nK䂻?!?/?Ry?O@)?L!u;??1l??w?J@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeata?????!8?
?]2@)?d??7i??1y????D.@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap/??[<???!?Fi?3+@)Pr?Md???1???M?%@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2vi????!k?k??#@)vi????1k?k??#@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor6w??\?v?!?'SN>?	@)6w??\?v?1?'SN>?	@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice{?%9`Ws?!F],n?*@){?%9`Ws?1F],n?*@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??'I?L??!!)[??\A@)/?
ҌEs?1,?A?@:Preprocessing2F
Iterator::Model????kz??!pkR??QP@)??Co??n?1?uZ\??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9NyF?@I6?}/?YX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	КiQ??КiQ??!КiQ??      ??!       "      ??!       *      ??!       2	;R}?E@;R}?E@!;R}?E@:      ??!       B      ??!       J	?M*k???M*k??!?M*k??R      ??!       Z	?M*k???M*k??!?M*k??b      ??!       JCPU_ONLYYNyF?@b q6?}/?YX@