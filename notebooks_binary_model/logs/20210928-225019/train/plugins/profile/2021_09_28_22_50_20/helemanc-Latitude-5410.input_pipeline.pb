	Zbe4?i@Zbe4?i@!Zbe4?i@	1??':@1??':@!1??':@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Zbe4?i@>??WXp??A@??$b@YmY?.???*	?l????`@2K
Iterator::Model::Map3??xy??!------P@)???6??1;???/7J@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?)??s??!??1@)O?\?	??1>????,@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?c???!???Ѫ?(@)?c???1???Ѫ?(@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?|?.PR??!????~'@)6=((E+??1Ӌӧ&? @:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::ZipYk(?Ѧ?!6?ip>l@@)?=~os?1N?&*l?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?wak??r?!??m??F@)?wak??r?1??m??F@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5z5@i?q?!y8NPk	@)5z5@i?q?1y8NPk	@:Preprocessing2F
Iterator::Model ?={.S??!e????P@)?o+?6k?1?f?Ss?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no91??':@I?r??/FX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>??WXp??>??WXp??!>??WXp??      ??!       "      ??!       *      ??!       2	@??$b@@??$b@!@??$b@:      ??!       B      ??!       J	mY?.???mY?.???!mY?.???R      ??!       Z	mY?.???mY?.???!mY?.???b      ??!       JCPU_ONLYY1??':@b q?r??/FX@