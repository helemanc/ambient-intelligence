	?aQW@?aQW@!?aQW@	:???E?@:???E?@!:???E?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?aQW@?;???Af/?N[?@Y??;3?p??*	?????i@2K
Iterator::Model::Map??m??!?W?egL@)$~?.r??1T??1a?F@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat4??!Y??Kv8@)C?5v????1d?@b5@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2m8,????!?<?I&@)m8,????1?<?I&@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?_[??g??!2? ?,@)'1?Z??1r??H??#@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?\?&???!?rp%?@)?\?&???1?rp%?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip0+?~N??!?????D@)?C4???y?1?N???7	@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???'*v?!???N??@)???'*v?1???N??@:Preprocessing2F
Iterator::Model`?eM,???!]?1 6M@)%???wj?1?;??M???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9:???E?@I?ЭJX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?;????;???!?;???      ??!       "      ??!       *      ??!       2	f/?N[?@f/?N[?@!f/?N[?@:      ??!       B      ??!       J	??;3?p????;3?p??!??;3?p??R      ??!       Z	??;3?p????;3?p??!??;3?p??b      ??!       JCPU_ONLYY:???E?@b q?ЭJX@