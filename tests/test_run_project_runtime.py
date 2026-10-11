from pathlib import Path
import unittest
from unittest.mock import patch

import run_project


class ProjectRuntimeCommandsTest(unittest.TestCase):
    def invoke_mode(self, arguments):
        target = run_project.ROOT/'outputs/test_mock_command'
        with patch.object(run_project,'invoke') as invoke, \
             patch.object(run_project,'new_output',return_value=target), \
             patch.object(run_project,'bootstrap'), patch.object(run_project.os,'chdir'):
            run_project.main(arguments)
            return invoke.call_args, target

    def test_steps_passes_explicit_device_and_source_directory(self):
        call,target = self.invoke_mode(['steps','--device','cuda:0','--source-dir','/tmp/source'])
        self.assertEqual(call.args,('scripts.run_step_counting','--output-csv',target/'penalty_step_counts.csv',
                         '--diagnostics-json',target/'step_diagnostics.json','--device','cuda:0',
                         '--clips-dir',Path('/tmp/source')))

    def test_camera_passes_runtime(self):
        call,_ = self.invoke_mode(['camera','--device','cpu','--source-dir','/tmp/source'])
        self.assertIn('--device',call.args)
        self.assertEqual(call.args[call.args.index('--device')+1],'cpu')
        self.assertEqual(call.args[call.args.index('--source-dir')+1],Path('/tmp/source'))

    def test_final_results_calls_builder_without_gpu_inference(self):
        call,target = self.invoke_mode(['final-results'])
        self.assertEqual(call.args,('scripts.build_final_results','--root',run_project.ROOT,'--output-dir',target))

    def test_eda_passes_feature_csv_and_output_directory(self):
        call,target = self.invoke_mode(['eda','--features-csv','/tmp/features.csv'])
        self.assertEqual(call.args,('src.eda_analysis','--features-csv',Path('/tmp/features.csv'),'--output-dir',target))

    def test_refresh_parser_exposes_upstream_directories(self):
        args = run_project.build_parser().parse_args(['refresh-ball','--device','cuda:0',
                         '--ball-cache-dir','/tmp/detections','--track-dir','/tmp/tracks',
                         '--ball-input-dir','/tmp/input','--source-dir','/tmp/source','--ffmpeg','/bin/ffmpeg'])
        self.assertEqual(args.ball_cache_dir,Path('/tmp/detections'))
        self.assertEqual(args.track_dir,Path('/tmp/tracks'))
        self.assertEqual(args.ball_input_dir,Path('/tmp/input'))
        self.assertEqual(args.source_dir,Path('/tmp/source'))


if __name__ == '__main__':
    unittest.main()
