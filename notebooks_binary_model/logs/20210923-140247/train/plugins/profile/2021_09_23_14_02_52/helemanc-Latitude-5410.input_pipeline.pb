	n½2o?@n½2o?@!n½2o?@	?hH?|?@?hH?|?@!?hH?|?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$n½2o?@ۤ???w??A????m@YX9??v???*	Zd;??e@2K
Iterator::Model::Map4??????!?i???Q@)=?E~???1?}@N@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatDP5z5@??!ǲHd9,@)???;۔?1Х?vO'@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?s`9B??!rL?S%$@)?s`9B??1rL?S%$@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap9????	??!gib?G%@)?J?8???1	??؟@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?Pj/??x?!?]r#??@)?Pj/??x?1?]r#??@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?j??P???!?K???;@)?Y.??s?1??_?L@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor+?&?|?q?!芋?*?@)+?&?|?q?1芋?*?@:Preprocessing2F
Iterator::Modelh??|?5??!?;혖R@)?????j?14?d+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?hH?|?@I????HX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ۤ???w??ۤ???w??!ۤ???w??      ??!       "      ??!       *      ??!       2	????m@????m@!????m@:      ??!       B      ??!       J	X9??v???X9??v???!X9??v???R      ??!       Z	X9??v???X9??v???!X9??v???b      ??!       JCPU_ONLYY?hH?|?@b q????HX@