	????/@????/@!????/@	?
a?)@?
a?)@!?
a?)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????/@+P??ô??A,??e@Y]???2???*	)\???Dd@2K
Iterator::Model::Map?'???I??!(}~oP@)"???/??1?)??K?J@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2 B\9{g??!m????(@) B\9{g??1m????(@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap??@????!U?W?.@)??R$_	??1?Z :"(@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?????y??!??v?F,@)?????m??1N[&?f'@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?mr??s?!?	^?K?@)?mr??s?1?	^?K?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip*??F????!??Y?D@@)???U+s?1?????@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3???/p?!?^n>U@)3???/p?1?^n>U@:Preprocessing2F
Iterator::ModelX zR&5??!?	ӿ??P@)??P?lm?1???*??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?
a?)@IX??f??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+P??ô??+P??ô??!+P??ô??      ??!       "      ??!       *      ??!       2	,??e@,??e@!,??e@:      ??!       B      ??!       J	]???2???]???2???!]???2???R      ??!       Z	]???2???]???2???!]???2???b      ??!       JCPU_ONLYY?
a?)@b qX??f??W@