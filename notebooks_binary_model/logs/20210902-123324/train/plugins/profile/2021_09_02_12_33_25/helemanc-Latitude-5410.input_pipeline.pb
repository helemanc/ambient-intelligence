	HO?C?-	@HO?C?-	@!HO?C?-	@	+? ??@+? ??@!+? ??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$HO?C?-	@#?GG???A???[?@Y??8?j???*	fffff>g@2K
Iterator::Model::Map????cw??!?хXR@)?i??%??1HN^?O@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2T????#??!̏1P?$@)T????#??1̏1P?$@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?v?
?ݖ?!?m?J?(@)??M?t??1{????b#@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap???g?R??!z??-X%@)?RB??^??1i&?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceh?N???t?!???z?@)h?N???t?1???z?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip????!?6?-?9@)?.???ur?1???o?c@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX??C?q?!? ?,??@)X??C?q?1? ?,??@:Preprocessing2F
Iterator::Modelw?4E????!_??tE?R@)??5"g?1??/??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s3.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9+? ??@IP????W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	#?GG???#?GG???!#?GG???      ??!       "      ??!       *      ??!       2	???[?@???[?@!???[?@:      ??!       B      ??!       J	??8?j?????8?j???!??8?j???R      ??!       Z	??8?j?????8?j???!??8?j???b      ??!       JCPU_ONLYY+? ??@b qP????W@