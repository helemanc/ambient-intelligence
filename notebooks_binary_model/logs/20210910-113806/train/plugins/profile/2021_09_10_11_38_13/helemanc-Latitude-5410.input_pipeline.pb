	堉?6?9@堉?6?9@!堉?6?9@	R?&?@R?&?@!R?&?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$堉?6?9@腐鐚(??A?JC@Y韆/???*	-矟颣~@2K
Iterator::Model::Map 歽rM佉?!h	梠鲆M@)?)U㈧-??1k??爡僄@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat宥}忷??!燫壡㏑3@)姄????1?@O6?0@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2)x
筊??!?]=?=)@))x
筊??1?]=?=)@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap䴕琎??!?&	~+@)滜)扦?1?(?]? @:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceY2球畾?!_鼤鰷?@)Y2球畾?1_鼤鰷?@:Preprocessing2F
Iterator::ModelV,~SX┯?!d?癘@)`#I畝??1畯鶑跹@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip(況厀蛊?!淽i7闛B@)篖M?7?1瞑&N@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor覌邶*绹?!嵃脹#@)覌邶*绹?1嵃脹#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9R?&?@I.逬?-怶@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	腐鐚(??腐鐚(??!腐鐚(??      ??!       "      ??!       *      ??!       2	?JC@?JC@!?JC@:      ??!       B      ??!       J	韆/???韆/???!韆/???R      ??!       Z	韆/???韆/???!韆/???b      ??!       JCPU_ONLYYR?&?@b q.逬?-怶@