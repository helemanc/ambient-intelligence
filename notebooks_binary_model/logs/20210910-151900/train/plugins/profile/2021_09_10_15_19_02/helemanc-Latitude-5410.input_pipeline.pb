	?x?&1@?x?&1@!?x?&1@	?H摜@?H摜@!?H摜@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?x?&1@v?|?H???A??מY?@Y??w??x??*	n???%c@2K
Iterator::Model::Map[닄????!??"?FN@)?J
,???1L??jPF@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2???)r??!????n+/@)???)r??1????n+/@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?k?????!?<?øt2@)2t????1??R?5k-@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?>???!?????V/@)?ᔹ?F??1?ᘽ??)@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zips?????!Q?m?B@)?R????w?1mqU??_@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceG??ǁw?!??fC??@)G??ǁw?1??fC??@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensort??Y5q?!?to???@)t??Y5q?1?to???@:Preprocessing2F
Iterator::Model??$??W??!?Z???	O@)F?Sweg?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?H摜@Is??6??W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v?|?H???v?|?H???!v?|?H???      ??!       "      ??!       *      ??!       2	??מY?@??מY?@!??מY?@:      ??!       B      ??!       J	??w??x????w??x??!??w??x??R      ??!       Z	??w??x????w??x??!??w??x??b      ??!       JCPU_ONLYY?H摜@b qs??6??W@