	Tr3?@Tr3?@!Tr3?@	yȪa?
@yȪa?
@!yȪa?
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Tr3?@?S???\??A?"1A@Y????P???*	??ʡEs@2K
Iterator::Model::Map???n???!"?? ?L@)r?#D??1??oH@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap???|y??!y???43@)?o%;6??1?'??0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat^=?1X??!n????80@)0?70?Q??1Д?k?*@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2:vP???!L?l??!@):vP???1L?l??!@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?uoEb???!Ns??2?C@)a?xwd???1??+?q@:Preprocessing2F
Iterator::Model???|????!??O
?xN@) ??Ud??1Y??
@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?? ???!1H??v?	@)?? ???11H??v?	@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?x???!^?=܌?	@)?x???1^?=܌?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9yȪa?
@I?????+X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S???\???S???\??!?S???\??      ??!       "      ??!       *      ??!       2	?"1A@?"1A@!?"1A@:      ??!       B      ??!       J	????P???????P???!????P???R      ??!       Z	????P???????P???!????P???b      ??!       JCPU_ONLYYyȪa?
@b q?????+X@