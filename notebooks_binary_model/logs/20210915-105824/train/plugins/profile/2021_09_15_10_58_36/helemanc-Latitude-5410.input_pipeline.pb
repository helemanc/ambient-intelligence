	+????@+????@!+????@	x?$l?|@x?$l?|@!x?$l?|@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+????@???Li???A	????@Y;?? ?>??*	+??·c@2K
Iterator::Model::Mapz8???n??!)??ګJM@)p]1#???1??J?c+F@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatMI???*??![?]??u:@)?|A??1?5?yGN5@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2q!??Fʖ?!6?sJ },@)q!??Fʖ?16?sJ },@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMappw?n?Ќ?!??Px?"@)?	?s3??1?[@@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Д?~??![ۺ?x?@)??Д?~??1[ۺ?x?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zipq??Ů?!m?t??;C@)?O?I?5s?1?dK?@:Preprocessing2F
Iterator::Model.?R????!?p?T,?N@)??̔??r?1?V;??@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?۠?[;q?!?,?)i?@)?۠?[;q?1?,?)i?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9x?$l?|@I?۞?lX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Li??????Li???!???Li???      ??!       "      ??!       *      ??!       2		????@	????@!	????@:      ??!       B      ??!       J	;?? ?>??;?? ?>??!;?? ?>??R      ??!       Z	;?? ?>??;?? ?>??!;?? ?>??b      ??!       JCPU_ONLYYx?$l?|@b q?۞?lX@