	"???ӏ@"???ӏ@!"???ӏ@	?~lq3@?~lq3@!?~lq3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"???ӏ@?HLP÷??A??9y??@Y?'d?ml??*	Zd;?OYk@2K
Iterator::Model::Map???????!??N,̉P@)?_?+?۾?1?ʈu?K@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat??lY?.??!?Mq??C8@) C?*??1<?h?v6@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2J)???Ƙ?!??S??&@)J)???Ƙ?1??S??&@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap.??H??!jϫ?A?@)?!??|?1?"B?q	@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?Y?e0??!+vR,?<@@)?Ye???v?1+TE? ?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice<P?<?v?!|Jt?@)<P?<?v?1|Jt?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorK?|%p?!????????)K?|%p?1????????:Preprocessing2F
Iterator::Model)?'?$???!??????P@)3?FY??h?1??a????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?~lq3@I
?tdX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?HLP÷???HLP÷??!?HLP÷??      ??!       "      ??!       *      ??!       2	??9y??@??9y??@!??9y??@:      ??!       B      ??!       J	?'d?ml???'d?ml??!?'d?ml??R      ??!       Z	?'d?ml???'d?ml??!?'d?ml??b      ??!       JCPU_ONLYY?~lq3@b q
?tdX@