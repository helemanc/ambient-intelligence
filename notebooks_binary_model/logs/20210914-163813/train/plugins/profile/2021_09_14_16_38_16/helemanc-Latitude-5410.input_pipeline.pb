	?X?_"~@?X?_"~@!?X?_"~@	??O??@??O??@!??O??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?X?_"~@-!?lV??A???7/?@Y {?\???*	Xd;?Oef@2K
Iterator::Model::Map?O?Y????!*??BJ@)?:u??<??1??????D@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?K?1?=??!1????<@)??J?????1|	??8@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat/ܹ0ҋ??!Iq+BC?,@)bJ$??(??1K>??((@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2`V(?????!ϸ?jJ$@)`V(?????1ϸ?jJ$@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??ݓ??z?!]?,g?@)??ݓ??z?1]?,g?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip#1?0&??!J餬G@)!??q4Gv?1??/?-I@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor4GV~?q?!????? @)4GV~?q?1????? @:Preprocessing2F
Iterator::ModelB]¡???!?[S??J@)g???uj?1???B???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??O??@I?`???PX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-!?lV??-!?lV??!-!?lV??      ??!       "      ??!       *      ??!       2	???7/?@???7/?@!???7/?@:      ??!       B      ??!       J	 {?\??? {?\???! {?\???R      ??!       Z	 {?\??? {?\???! {?\???b      ??!       JCPU_ONLYY??O??@b q?`???PX@