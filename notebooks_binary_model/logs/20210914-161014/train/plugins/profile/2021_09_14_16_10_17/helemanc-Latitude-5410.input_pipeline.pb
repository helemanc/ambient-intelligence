	?2Wu	@?2Wu	@!?2Wu	@	??B?=@??B?=@!??B?=@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?2Wu	@a???????A???uS?@Y??n????*	??x?&b@2K
Iterator::Model::Map???e????!?2??~?P@)|V?j-??1BW^K\K@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???|?R??!?|z??/@)??N????1޾ 2??)@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??&S??!r9?2??)@)??&S??1r9?2??)@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap????A??!????R
&@)F?Swe??1??????@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipN} y?P??!?mB??B>@)$H???8t?1?~??k@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceV??W9r?!a?)???@)V??W9r?1a?)???@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;S???.q?!??f%?L@);S???.q?1??f%?L@:Preprocessing2F
Iterator::Modele?X???!?d?ToQ@)6??`?
i?176F??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??B?=@I??VX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a???????a???????!a???????      ??!       "      ??!       *      ??!       2	???uS?@???uS?@!???uS?@:      ??!       B      ??!       J	??n??????n????!??n????R      ??!       Z	??n??????n????!??n????b      ??!       JCPU_ONLYY??B?=@b q??VX@