import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import lab_experiment


class LabExperimentTest(unittest.TestCase):
    def test_ssh_remote_arguments_are_shell_quoted_and_script_uses_stdin(self):
        value = 'spaces $(untrusted) ; literal'
        with patch.object(lab_experiment.subprocess, 'run') as run:
            lab_experiment.remote_python('lab4090', '/path with spaces/python', 'print(42)', value)
        args, kwargs = run.call_args
        self.assertEqual(shlex.split(args[0][-1]), ['/path with spaces/python', '-', value])
        self.assertEqual(kwargs['input'], 'print(42)')
        self.assertTrue(kwargs['check'])
        self.assertNotIn('shell', kwargs)

    def test_start_forwards_options_after_separator_without_interpreting_them(self):
        with patch.object(lab_experiment, 'remote_python') as remote:
            lab_experiment.main(['start', '--module', 'scripts.run_step_counting',
                                 '--job-id', 'steps-01', '--', '--device', 'cuda:0',
                                 '--model', '/models/pose with spaces.pt'])
        call = remote.call_args
        self.assertEqual(call.args[4], 'scripts.run_step_counting')
        self.assertEqual(json.loads(call.args[5]), ['--device', 'cuda:0', '--model', '/models/pose with spaces.pt'])
        self.assertEqual(call.args[6], 'steps-01')

    def test_rejects_remote_traversal_and_unsafe_identifiers_before_ssh(self):
        cases = [
            ['status', '--ssh-alias=-x'],
            ['status', '--remote-base', '/tmp/../etc'],
            ['start', '--module', 'scripts.foo', '--job-id', '..'],
            ['start', '--module', 'scripts.', '--job-id', 'job'],
            ['start', '--module', 'scripts.foo', '--job-id', 'job', '--memory-fraction', '.9'],
            ['fetch', '--remote-artifact', '../elsewhere', '--local-artifact', '/tmp/output'],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.object(lab_experiment, 'remote_python') as remote, \
                    patch.object(lab_experiment.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    lab_experiment.main(arguments)
                remote.assert_not_called()
                run.assert_not_called()

    def test_status_accepts_package_messages_before_runtime_json(self):
        value = {'cuda_available': True, 'input_files': [], 'new_source_video_inference': False}
        result = subprocess.CompletedProcess([], 0, stdout='model startup message\n__LAB_RUNTIME_JSON__'+json.dumps(value))
        with tempfile.TemporaryDirectory() as temporary, patch.object(lab_experiment, 'remote_python', return_value=result):
            destination = Path(temporary)/'runtime.json'
            lab_experiment.main(['status', '--runtime-output', str(destination)])
            self.assertEqual(json.loads(destination.read_text()), value)

    def test_remote_wrappers_compile_on_supported_python(self):
        for name in ('REMOTE_PREPARE', 'REMOTE_STATUS', 'REMOTE_START'):
            with self.subTest(name=name):
                compile(getattr(lab_experiment, name), name, 'exec')


if __name__ == '__main__':
    unittest.main()
