"""Automatic output tests, runnable with unittest without extra test dependencies."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from motionguard.cli import main,training_destination


class AutoOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.original=Path.cwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.original)
        self.tmp.cleanup()

    def check_command(self,command,extension):
        def fit(manifest,out,cfg):
            Path(out).write_bytes(b'checkpoint fixture')
        with patch('motionguard.train.fit_tcn',side_effect=fit), patch('motionguard.train.fit_baseline',side_effect=fit), patch('motionguard.train.fit_cnn_lstm',side_effect=fit), patch('builtins.print') as output:
            main([command,'--manifest','input.jsonl'])
        paths=list(Path('runs').rglob('model'+extension))
        self.assertEqual(len(paths),1)
        self.assertEqual(paths[0].read_bytes(),b'checkpoint fixture')
        self.assertIn(str(paths[0].resolve()),' '.join(call.args[0] for call in output.call_args_list))
        self.assertNotEqual(training_destination(command).parent,paths[0].resolve().parent)

    def test_tcn_automatic_path(self):
        self.check_command('train-tcn','.pt')

    def test_cnn_lstm_automatic_path(self):
        self.check_command('train-cnn-lstm','.pt')

    def test_baseline_automatic_path(self):
        self.check_command('train-baseline','.joblib')

    def test_explicit_checkpoint_path_preserved(self):
        self.assertEqual(training_destination('train-tcn','custom/test.pt'),Path.cwd()/'custom'/'test.pt')
        self.assertFalse(Path('runs').exists())

if __name__=='__main__': unittest.main()
