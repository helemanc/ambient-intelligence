	5?+-#?@5?+-#?@!5?+-#?@	???7?@???7?@!???7?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$5?+-#?@]??վ?A>U?W?@YX?f,????*F?z??h@)       =2K
Iterator::Model::MapG6u??!@??NUN@)	l??3???1&%?Z?I@:Preprocessing2F
Iterator::Model?L!u??!@ʬ?`S@)wj.7???1?F;`?0@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?c?????!??????(@)?XİØ??1빝60?$@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?k?,	P??!i`??;#@)?k?,	P??1i`??;#@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?'eRC??!??? d
 @)?Q???T??1Mo5?>@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicer????u?!???I?@)r????u?1???I?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip ?O????!??L?}6@)?5Y??q?1???ॏ@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorP?}:3p?!???%" @)P?}:3p?1???%" @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???7?@I??"BFPX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]??վ?]??վ?!]??վ?      ??!       "      ??!       *      ??!       2	>U?W?@>U?W?@!>U?W?@:      ??!       B      ??!       J	X?f,????X?f,????!X?f,????R      ??!       Z	X?f,????X?f,????!X?f,????b      ??!       JCPU_ONLYY???7?@b q??"BFPX@