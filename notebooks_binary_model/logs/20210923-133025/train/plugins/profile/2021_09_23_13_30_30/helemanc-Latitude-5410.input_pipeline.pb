	???Rz@???Rz@!???Rz@	? ?B????? ?B????!? ?B????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Rz@?? ??Ե?A??(&o?@Y?R$_	???*	?"??~?a@2K
Iterator::Model::Map5@i?QH??!?YwxN@))#. ?ұ?1d?"_H@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeato?????!3?c???0@)?I??{??1q=???,@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?mē?̘?!?gg+??0@)?,??????1??B???*@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV22t??ב?!$.ԟQe(@)2t??ב?1$.ԟQe(@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceC?l??t?!^?/?@)C?l??t?1^?/?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipGW??:??!eEC;B?B@)????t?1?ߝ4o@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??t?i?p?!?Q?V?@)??t?i?p?1?Q?V?@:Preprocessing2F
Iterator::ModelZH?????!???ĽaO@)?:???Re?1?^s,?(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9? ?B????I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? ??Ե??? ??Ե?!?? ??Ե?      ??!       "      ??!       *      ??!       2	??(&o?@??(&o?@!??(&o?@:      ??!       B      ??!       J	?R$_	????R$_	???!?R$_	???R      ??!       Z	?R$_	????R$_	???!?R$_	???b      ??!       JCPU_ONLYY? ?B????b q?????X@