	???o
?@???o
?@!???o
?@	?u"?{@?u"?{@!?u"?{@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???o
?@H?Ȱ?7??A??q@Y??fd????*	X9?Ȅp@2K
Iterator::Model::Mapeq??????!???탙Q@)?Ϛ??1??kD??L@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?۽?'G??!??/[F?)@)?۽?'G??1??/[F?)@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatt?f?????!?Ƶ>??+@)???w??1~"?]?&@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?wb֋???!4i(%@)Va3?ْ?1!g8?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?,^,???!??G4?@)?,^,???1??G4?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipF????Ų?!???s?;@)?4?8EG??1??¾?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?/עh{?!֑˸@@)?/עh{?1֑˸@@:Preprocessing2F
Iterator::Model??9??q??!??cR@)?	L?ut?1^??ŷ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?u"?{@I??E?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?Ȱ?7??H?Ȱ?7??!H?Ȱ?7??      ??!       "      ??!       *      ??!       2	??q@??q@!??q@:      ??!       B      ??!       J	??fd??????fd????!??fd????R      ??!       Z	??fd??????fd????!??fd????b      ??!       JCPU_ONLYY?u"?{@b q??E?W@