	%"????@%"????@!%"????@	??D$?R@??D$?R@!??D$?R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$%"????@V?pA???AgC??A?@Y?k???P??*	,???Cf@2K
Iterator::Model::Mapo??;????!?[H???R@)@1?d????1n???~eL@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2:?%???!??E?1@):?%???1??E?1@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat|*?=%???!?????&@)?yUg???1?B'KR"@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap*???K??! ?d??$@)U???*È?1F?!'@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???1??w?!siO}4?	@)???1??w?1siO}4?	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipU1?~?٥?!i????7@)L5??r?1q?죿@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????p?!?[w?e@)??????p?1?[w?e@:Preprocessing2F
Iterator::ModelV?@?)V??!&??;?S@)?y?'Lh?1N!??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??D$?R@I\?ݶk=X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V?pA???V?pA???!V?pA???      ??!       "      ??!       *      ??!       2	gC??A?@gC??A?@!gC??A?@:      ??!       B      ??!       J	?k???P???k???P??!?k???P??R      ??!       Z	?k???P???k???P??!?k???P??b      ??!       JCPU_ONLYY??D$?R@b q\?ݶk=X@