	??(?[z@??(?[z@!??(?[z@	??+???@??+???@!??+???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??(?[z@?????խ?A?6?X+@Y??w???*	+????_@2K
Iterator::Model::Map?o?^}<??!8}?:??N@)m7?7M???14i}P??F@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?P?[???!(???0@)?P?[???1(???0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?c?Cԗ?!(??.2@)	??g????1?.\?.d.@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?r?4???!??s???,@)??U?3??1V.?s??%@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicet#,*?tr?!l???)@)t#,*?tr?1l???)@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip? ?b??!?@΁?A@)	N} y?p?1????Z?	@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?VC?Ko?!??@?$?@)?VC?Ko?1??@?$?@:Preprocessing2F
Iterator::ModelUg????!?_??P@)???<,?j?1"?px@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??+???@I?>Ⱥ#X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????խ??????խ?!?????խ?      ??!       "      ??!       *      ??!       2	?6?X+@?6?X+@!?6?X+@:      ??!       B      ??!       J	??w?????w???!??w???R      ??!       Z	??w?????w???!??w???b      ??!       JCPU_ONLYY??+???@b q?>Ⱥ#X@