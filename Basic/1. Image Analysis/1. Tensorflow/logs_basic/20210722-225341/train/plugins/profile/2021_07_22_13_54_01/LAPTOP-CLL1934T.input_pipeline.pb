	??#)?}a@??#)?}a@!??#)?}a@	{r/ښ=??{r/ښ=??!{r/ښ=??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??#)?}a@ D2??z??A???Yqa@Y?B]???rEagerKernelExecute 0*	?Zd(?@2?
TIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::MapAndBatch::ParallelMapV2 @ٔ+?}C@!??f"+?X@)@ٔ+?}C@1??f"+?X@:Preprocessing2?
aIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::MapAndBatch::ParallelMapV2::TensorSlice {/?h???!٨?????){/?h???1٨?????:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::MapAndBatch?`?unڜ?!Ӄ˽?b??)?`?unڜ?1Ӄ˽?b??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat??Q??Z??!???(???)?*???ڗ?1?;'?f??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???}????!-_~?]y??)?{+Ԁ?1?Mmr??:Preprocessing2F
Iterator::Model???t????!??TE???)֪]?z?1?n?J<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9{r/ښ=??I#tI???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 D2??z?? D2??z??! D2??z??      ??!       "      ??!       *      ??!       2	???Yqa@???Yqa@!???Yqa@:      ??!       B      ??!       J	?B]????B]???!?B]???R      ??!       Z	?B]????B]???!?B]???b      ??!       JCPU_ONLYY{r/ښ=??b q#tI???X@