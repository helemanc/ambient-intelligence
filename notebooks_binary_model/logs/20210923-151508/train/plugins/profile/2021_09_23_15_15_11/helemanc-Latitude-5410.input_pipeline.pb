	7U?q@7U?q@!7U?q@	ݽ(???@ݽ(???@!ݽ(???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$7U?q@Ֆ:?????A?$???@Y ???Qc??*	???K7An@2K
Iterator::Model::Map?;??J"??!???e&HT@)ލ?A???1}??&i<R@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?o?DIH??!%k???] @)?o?DIH??1%k???] @:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat???n??!?%??-?"@)??ӝ'???1%-??@?@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap???c> ??!	???@)????t!??1?
???@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??>t?!?0G??U @)??>t?1?0G??U @:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?? ????!??+??1@)N?G??q?1?|?$???:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???g?n?!?z??k???)???g?n?1?z??k???:Preprocessing2F
Iterator::Model??0_^???!u??T@)c	kc??g?1t?!????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ݽ(???@I"t? ?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ֆ:?????Ֆ:?????!Ֆ:?????      ??!       "      ??!       *      ??!       2	?$???@?$???@!?$???@:      ??!       B      ??!       J	 ???Qc?? ???Qc??! ???Qc??R      ??!       Z	 ???Qc?? ???Qc??! ???Qc??b      ??!       JCPU_ONLYYݽ(???@b q"t? ?W@