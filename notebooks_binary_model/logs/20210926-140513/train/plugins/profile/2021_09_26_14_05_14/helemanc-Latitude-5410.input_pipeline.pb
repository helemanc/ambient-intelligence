	?ȳ?	@?ȳ?	@!?ȳ?	@	?S}s?w@?S}s?w@!?S}s?w@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ȳ?	@⏢??C??A???%V&@Y	6??g??*	>
ףp)a@2K
Iterator::Model::Map?Jw?ِ??!?Zo??P@)?"0?70??1?S?r?sH@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2s?????!+?><%2@)s?????1+?><%2@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatT?:???!h?Όk0@)?g?,{??1f[<tJ*@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?[[%X??!??~&5)$@)=ڨN??1??Ax??@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice0?Qd??t?!??v??Y@)0?Qd??t?1??v??Y@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip\??????!?ě???=@)?JU?r?1????i?
@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??g?ejr?!?U?'?2
@)??g?ejr?1?U?'?2
@:Preprocessing2F
Iterator::ModelV,~SX???!??ښ?Q@)?n???q?1??6Mb?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?S}s?w@I`dBX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	⏢??C??⏢??C??!⏢??C??      ??!       "      ??!       *      ??!       2	???%V&@???%V&@!???%V&@:      ??!       B      ??!       J		6??g??	6??g??!	6??g??R      ??!       Z		6??g??	6??g??!	6??g??b      ??!       JCPU_ONLYY?S}s?w@b q`dBX@