	^?Y-?G@^?Y-?G@!^?Y-?G@	?]????-@?]????-@!?]????-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^?Y-?G@CƣT???A?ӝ'?c@Yō[??M??*	?Q??f?@2K
Iterator::Model::Map??????!D~??-?W@)??5[yI??1**(??IW@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2HO?C?͙?!H??ͼ4@)HO?C?͙?1H??ͼ4@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatL?Ƽ?8??!?r???)l_@/ܹ??1j4?????:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?t?? ??!uç?%+??)?? {??1??غ????:Preprocessing2F
Iterator::Modelt?Y?b+??!>6`?X@)0?x??n?13?_?Je??:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice<??~Kq?!??<+@???)<??~Kq?1??<+@???:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip.=??????!?>8???@)?????p?1??bT???:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/3l??k?!???u???)/3l??k?1???u???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?]????-@IE nNU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	CƣT???CƣT???!CƣT???      ??!       "      ??!       *      ??!       2	?ӝ'?c@?ӝ'?c@!?ӝ'?c@:      ??!       B      ??!       J	ō[??M??ō[??M??!ō[??M??R      ??!       Z	ō[??M??ō[??M??!ō[??M??b      ??!       JCPU_ONLYY?]????-@b qE nNU@