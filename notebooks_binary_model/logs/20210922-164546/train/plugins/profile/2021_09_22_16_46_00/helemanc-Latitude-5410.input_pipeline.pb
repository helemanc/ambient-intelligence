	??=??@??=??@!??=??@	?j??M6@?j??M6@!?j??M6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??=??@??	?Yپ?A?????@Y??^?s???*	K7?A`??@2K
Iterator::Model::Map?wJk??!?????1X@)q?i???1??K??P@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2zm6Vb???!??A?[>@)zm6Vb???1??A?[>@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeate?uʓ?!?=?s??)??bFx{??1?xs?????:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapՒ?r0???!???̌???)???4cф?1???/???:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice>?4a??x?!??T?_??)>?4a??x?1??T?_??:Preprocessing2F
Iterator::Model????y??!NT?5CX@)???8m?1????P??:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?4ӽN???!Hv5]?@)???*?wk?1?p?8F??:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%???wj?!)cS?C]??)%???wj?1)cS?C]??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?j??M6@IB?̏l~S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??	?Yپ???	?Yپ?!??	?Yپ?      ??!       "      ??!       *      ??!       2	?????@?????@!?????@:      ??!       B      ??!       J	??^?s?????^?s???!??^?s???R      ??!       Z	??^?s?????^?s???!??^?s???b      ??!       JCPU_ONLYY?j??M6@b qB?̏l~S@