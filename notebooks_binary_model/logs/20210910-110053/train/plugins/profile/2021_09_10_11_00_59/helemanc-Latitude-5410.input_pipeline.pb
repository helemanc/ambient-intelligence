	?+?S?@?+?S?@!?+?S?@	??SRnW
@??SRnW
@!??SRnW
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?+?S?@o??g??Au ?]??@Y??d??~??*	?Q???p@2K
Iterator::Model::Map(c|??l??! ????S@)?;ۤ???1;?s?)Q@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?;1??P??!`c?{&@)?;1??P??1`c?{&@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatU?W????!^?G?B#@)ZK ?)??1?0?#??@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap,??26t??!g?P?@@)3??(]??1 X??$@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceJ+?y?!?????7@)J+?y?1?????7@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??0Xru?!????%??)??0Xru?1????%??:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip????????!(??`?2@)??C?bt?1????4???:Preprocessing2F
Iterator::Model+???,??!6:\?']T@)Y?n??s?1?Ť`????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??SRnW
@Icbm?D-X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o??g??o??g??!o??g??      ??!       "      ??!       *      ??!       2	u ?]??@u ?]??@!u ?]??@:      ??!       B      ??!       J	??d??~????d??~??!??d??~??R      ??!       Z	??d??~????d??~??!??d??~??b      ??!       JCPU_ONLYY??SRnW
@b qcbm?D-X@