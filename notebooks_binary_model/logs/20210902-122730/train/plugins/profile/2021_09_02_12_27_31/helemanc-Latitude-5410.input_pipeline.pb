	$?????@$?????@!$?????@	?4F_@?4F_@!?4F_@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$?????@?Ŧ?B??A?e???@Y? ?bG??*	?Zd?b@2K
Iterator::Model::Map~?????!??{.R@).V?`???1
 ?M?tN@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?????!f??:)?-@)??`<??1?!W>??(@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2???,????!?}$???'@)???,????1?}$???'@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?˵hچ?!x????@)??c"?y?1??P??@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceR??m?t?!?.N?	@)R??m?t?1?.N?	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?????K??!<V?k?8@)???Fu:p?1 ????@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoro??m?n?!?D?c@)o??m?n?1?D?c@:Preprocessing2F
Iterator::Model;???????!q?=%?R@)?'??Ql?1?Ui?,U@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?4F_@I??ʿW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Ŧ?B???Ŧ?B??!?Ŧ?B??      ??!       "      ??!       *      ??!       2	?e???@?e???@!?e???@:      ??!       B      ??!       J	? ?bG??? ?bG??!? ?bG??R      ??!       Z	? ?bG??? ?bG??!? ?bG??b      ??!       JCPU_ONLYY?4F_@b q??ʿW@