	?5?!?@?5?!?@!?5?!?@	L?6d?@L?6d?@!L?6d?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?5?!?@C??3??A??ܵ??@YIط????*	????x?i@2K
Iterator::Model::MapS?K?^??!???O?GR@)u?l?????1?k?<??O@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?h?hs???!u!?*@)b??U???14???f$@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?&??鳓?!?e???"@)?&??鳓?1?e???"@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat%??????!A?:?u?$@)?,D????1?|C?? @:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??v?w?!%Ql??@)??v?w?1%Ql??@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??vۅ???!??d.d9@);S???.q?1r?? 8 @:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ek}?p?!\??=G??)??ek}?p?1\??=G??:Preprocessing2F
Iterator::Model???????!?8?f??R@)T?qs*i?1kH?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9M?6d?@I??K??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C??3??C??3??!C??3??      ??!       "      ??!       *      ??!       2	??ܵ??@??ܵ??@!??ܵ??@:      ??!       B      ??!       J	Iط????Iط????!Iط????R      ??!       Z	Iط????Iط????!Iط????b      ??!       JCPU_ONLYYM?6d?@b q??K??X@