	???G??@???G??@!???G??@	=T?S5,@=T?S5,@!=T?S5,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???G??@4??𽿩?A+?m???@Y;??u?+??*	??n??c@2K
Iterator::Model::Mapb??4?8??!?ynu??Q@)n??Ũ??1ur??M@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?<??@??!Mr?tZ+@)?<??@??1Mr?tZ+@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat????????!͏D??.@)p?܁:??1e??8*@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap*X?l:??!???@)TUh ??|?1?D??@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?Z^??6s?!P??c?@)?Z^??6s?1P??c?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip???????! Ҫ?#v9@)/n??r?1???+O'@:Preprocessing2F
Iterator::Model?????Q??!xKw?R@)?R??%?q?1z2???@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??m?!?Yp?"@)?g??m?1?Yp?"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9=T?S5,@I??ƪ<?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4??𽿩?4??𽿩?!4??𽿩?      ??!       "      ??!       *      ??!       2	+?m???@+?m???@!+?m???@:      ??!       B      ??!       J	;??u?+??;??u?+??!;??u?+??R      ??!       Z	;??u?+??;??u?+??!;??u?+??b      ??!       JCPU_ONLYY=T?S5,@b q??ƪ<?W@