import tensorflow as tf
import tfx
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import Transform
from tfx.components import Trainer
from tfx.proto import trainer_pb2
from tfx.components import Pusher
from tfx.proto import pusher_pb2
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import os

_data_path = #add path to directory which contains all the training files
_serving_dir = #add path to directory where you will be storing the final trained model
_num_train_steps = #add number
_num_eval_steps = #add number

_pipeline_name = #name the pipeline
_pipeline_root = #specify location of pipeline storage
_metadata_root = #specify location for sqlite db that will facilitate metadata storage
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=0',
]

output_config = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(
        splits = [
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ]
    )
)

example_gen = CsvExampleGen(input_base="/data_root", output_config=output_config)

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

schema_gen = SchemaGen(statistics = statistics_gen.outputs['statistics'], infer_feature_shape = True)

transform_file = 'transform.py'

transform = Transform(
    examples = example_gen.outputs['examples'],
    schema = schema_gen.outputs['schema'],
    module_file = transform_file
)

trainer_file = 'trainer_module.py'

trainer = Trainer(
    module_file = trainer_file,
    examples = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema = schema_gen.outputs['schema'],
    train_args = trainer_pb2.TrainArgs(num_steps = 100),
    eval_args = trainer_pb2.EvalArgs(num_steps = 10)
)

pusher = Pusher(
      model=trainer.outputs['model'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=_serving_dir)))

if __name__ == "__main__":
    pipeline_components = [
      example_gen,
      statistics_gen,
      schema_gen,
      transform,
      trainer,
      pusher
    ]

    tfx_pipeline = pipeline.Pipeline(
        pipeline_name = _pipeline_name,
        pipeline_root = _pipeline_root,
        components = pipeline_components,
        metadata_connection_config = metadata.sqlite_metadata_connection_config(metadata_path)
        enable_cache = False,
        beam_pipeline_args = _beam_pipeline_args
    )

    BeamDagRunner().run(tfx_pipeline)
