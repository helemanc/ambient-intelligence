	??N?`=@??N?`=@!??N?`=@	?M??3???M??3??!?M??3??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??N?`=@gC??A|??A???Дm@Y?o{??v??*	????Ɠh@2K
Iterator::Model::Map?a?????!??49p?D@)?Ȱ?72??1?R?0?>@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?N????!h?-??A@)AG?Z?Q??14b?H>@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap??WWj??!?m??kL1@)??I???1x?u?n*@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2??$??I??!{??|_'$@)??$??I??1{??|_'$@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@OI???![??x@)@OI???1[??x@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip{?V??׼?!?EԷ?L@)?X??L/??19W@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?N[#?q??!Pa???U@)?N[#?q??1Pa???U@:Preprocessing2F
Iterator::ModelHlw?}??!??+HYE@)????Gj?1_H?P???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?M??3??I??60?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	gC??A|??gC??A|??!gC??A|??      ??!       "      ??!       *      ??!       2	???Дm@???Дm@!???Дm@:      ??!       B      ??!       J	?o{??v???o{??v??!?o{??v??R      ??!       Z	?o{??v???o{??v??!?o{??v??b      ??!       JCPU_ONLYY?M??3??b q??60?X@