	??ؖ?@??ؖ?@!??ؖ?@	!r?힛@!r?힛@!!r?힛@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ؖ?@?jH?c???AA*Ŏ??@Y?!U????*	E????k@2K
Iterator::Model::Map?Ͻ???!?ɓ??H@)6=((E+??1??dM ?D@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip@?? kպ?!?U	?B-H@)??Ln??1kڙ???6@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?S:X????!و???v.@)?Q}>??1??xF?'@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatw?n??\??!??%@)???[???1O?$]E?!@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?*k??q??!?#&3jo@)?*k??q??1?#&3jo@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceVG?tF~?!sSS?F@)VG?tF~?1sSS?F@:Preprocessing2F
Iterator::Model????6???!u??;??I@){/?h?r?1Q?ς?L @:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'???Sn?!?IKw?R??)'???Sn?1?IKw?R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9!r?힛@Io?#cX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?jH?c????jH?c???!?jH?c???      ??!       "      ??!       *      ??!       2	A*Ŏ??@A*Ŏ??@!A*Ŏ??@:      ??!       B      ??!       J	?!U?????!U????!?!U????R      ??!       Z	?!U?????!U????!?!U????b      ??!       JCPU_ONLYY!r?힛@b qo?#cX@