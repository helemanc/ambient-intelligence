	?.R(?@?.R(?@!?.R(?@	??
?S@??
?S@!??
?S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?.R(?@???s?v??A?|?5^?@Y??b('???*	]???(Lm@2K
Iterator::Model::Map??Im ??!%???> T@)????????1??j?SSQ@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2???[???!?I?Yg%@)???[???1?I?Yg%@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?Z	?%q??!1>h??"@)? ???ے?1k??Vn@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap!?K????!y?*? @)h?
?O??1??Lo?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?=z?}?v?!?w?@)?=z?}?v?1?w?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?r۾G???!?ə??R2@)?&S?r?1Nň???:Preprocessing2F
Iterator::Model&5?؀??!???BkT@)?T?^p?1z:?????:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??2R??l?!?c??????)??2R??l?1?c??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??
?S@I??T???W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???s?v?????s?v??!???s?v??      ??!       "      ??!       *      ??!       2	?|?5^?@?|?5^?@!?|?5^?@:      ??!       B      ??!       J	??b('?????b('???!??b('???R      ??!       Z	??b('?????b('???!??b('???b      ??!       JCPU_ONLYY??
?S@b q??T???W@