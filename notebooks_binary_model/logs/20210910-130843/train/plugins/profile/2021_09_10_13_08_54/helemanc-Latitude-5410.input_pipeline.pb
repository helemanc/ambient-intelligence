	??D??@??D??@!??D??@	?E(d??@?E(d??@!?E(d??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??D??@B͐*?W??A???@Y-z?m???*	J+??d@2K
Iterator::Model::Map???v?Ӿ?!????|CR@)cFx{??1 )Efh?N@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?7??Ø?!??vX-@)s?w?????1`9Z$??(@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2}???E??!A??E?&@)}???E??1A??E?&@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?0
?Ƿ??!?{???@)Lo.2~?1?F????@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip$^?????!G?0w?v8@)?[?O?r?1{?@?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicek???t=q?!`jm@)k???t=q?1`jm@:Preprocessing2F
Iterator::Model??հ߿?!??3b_?R@)?????p?1?~H'O?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorO;?5Y?n?!?)??&@)O;?5Y?n?1?)??&@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?E(d??@I?{??ǡW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B͐*?W??B͐*?W??!B͐*?W??      ??!       "      ??!       *      ??!       2	???@???@!???@:      ??!       B      ??!       J	-z?m???-z?m???!-z?m???R      ??!       Z	-z?m???-z?m???!-z?m???b      ??!       JCPU_ONLYY?E(d??@b q?{??ǡW@