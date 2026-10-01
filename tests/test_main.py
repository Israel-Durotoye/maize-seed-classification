import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import main


def metadata():
    return {
        'schema_version': 2, 'class_names': main.CLASSES,
        'class_to_index': dict(zip(main.CLASSES, range(3))),
        'image_size': [2, 3], 'input_color_order': 'RGB', 'input_dtype': 'float32',
        'external_input_range': [0.0, 255.0], 'resize_method': 'pillow_bilinear',
        'decision_policy': {'actuation_enabled': True, 'empty_confidence': .9, 'empty_margin': .3,
                            'class_thresholds': {c: {'confidence': .8, 'margin': .2} for c in main.CLASSES[:2]}},
    }


class InferenceTests(unittest.TestCase):
    def test_all_three_classes_and_uncertainty(self):
        for probs, label, eligible, empty in [
            ([.96,.02,.02], 'BAD_SEED', True, False),
            ([.02,.96,.02], 'GOOD_SEED', True, False),
            ([.02,.02,.96], 'NO_MAIZE', False, True),
            ([.48,.49,.03], 'GOOD_SEED', False, False),
            ([.15,.10,.75], 'NO_MAIZE', False, False),
        ]:
            prediction = main.interpret_probabilities(probs, metadata())
            self.assertEqual((prediction.label, prediction.eligible, prediction.empty), (label, eligible, empty))

    def test_invalid_probabilities_fail_closed(self):
        for values in [[.5], [1,1,1], [-.1,.6,.5], [np.nan,0,1], [np.inf,0,0]]:
            with self.assertRaises(ValueError): main.interpret_probabilities(values, metadata())

    def test_disabled_and_invalid_policy(self):
        data = metadata()
        data['decision_policy']['actuation_enabled'] = False
        self.assertFalse(main.interpret_probabilities([0,1,0], data).eligible)
        main.validate_metadata(data)
        for key, value in [('class_names', ['GOOD_SEED','BAD_SEED','NO_MAIZE']),
                           ('input_color_order', 'BGR'), ('schema_version', 1), ('image_size', [-1,224])]:
            changed = copy.deepcopy(data); changed[key] = value
            with self.assertRaises(ValueError): main.validate_metadata(changed)
        data['decision_policy']['class_thresholds']['GOOD_SEED']['confidence'] = float('nan')
        with self.assertRaises(ValueError): main.validate_metadata(data)

    def test_rgb_range_size_and_exif(self):
        image = Image.new('RGB', (9, 6), (255, 20, 5))
        tensor = main.prepare_rgb(image, (2, 3))
        self.assertEqual(tensor.shape, (1,2,3,3))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0,0,0], [255,20,5])
        exif_image = Image.new('RGB', (3, 2))
        exif_image.putpixel((0,0), (255,0,0))
        exif_image.getexif()[274] = 6
        oriented = main.prepare_rgb(exif_image, (3,2))
        np.testing.assert_array_equal(oriented[0,0,1], [255,0,0])

    def test_classifier_checksum_and_tensor_contract(self):
        class FakeInterpreter:
            def __init__(self, **kwargs): pass
            def allocate_tensors(self): pass
            def get_input_details(self): return [{'shape':[1,2,3,3], 'dtype':np.float32, 'index':0}]
            def get_output_details(self): return [{'shape':[1,3], 'dtype':np.float32, 'index':1}]
            def set_tensor(self, index, value): self.tensor = value
            def invoke(self): pass
            def get_tensor(self, index): return np.array([[.96,.02,.02]], np.float32)
        package = types.ModuleType('ai_edge_litert')
        module = types.ModuleType('ai_edge_litert.interpreter'); module.Interpreter = FakeInterpreter
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'test.tflite'; model.write_bytes(b'test model')
            data = metadata(); data['tflite_sha256'] = {model.name:hashlib.sha256(model.read_bytes()).hexdigest()}
            config = Path(directory)/'deployment_metadata.json'; config.write_text(json.dumps(data))
            with patch.dict(sys.modules, {'ai_edge_litert':package, 'ai_edge_litert.interpreter':module}):
                classifier = main.Classifier(model, config)
                self.assertEqual(classifier.predict(Image.new('RGB',(10,10))).label, 'BAD_SEED')
                model.write_bytes(b'different model')
                with self.assertRaises(ValueError): main.Classifier(model, config)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.gate = main.DecisionGate(stable_frames=3, stable_seconds=.15, empty_frames=3, empty_seconds=.3, cooldown=.5)
        self.good = main.interpret_probabilities([.01,.98,.01], metadata())
        self.bad = main.interpret_probabilities([.98,.01,.01], metadata())
        self.empty = main.interpret_probabilities([.01,.01,.98], metadata())
        self.uncertain = main.interpret_probabilities([.45,.50,.05], metadata())

    def arm(self, start=0):
        for t in [start, start+.2, start+.4]: self.assertIsNone(self.gate.observe(self.empty,t))
        self.assertTrue(self.gate.armed)

    def test_starts_disarmed_and_one_command_per_seed(self):
        for t in [0,.1,.2]: self.assertIsNone(self.gate.observe(self.good,t))
        self.arm(.3)
        for t in [.8,.9]: self.assertIsNone(self.gate.observe(self.good,t))
        self.assertEqual(self.gate.observe(self.good,1.01),'GOOD_SEED')
        for t in [1.1,1.2,1.3,1.4]: self.assertIsNone(self.gate.observe(self.good,t))
        # One apparent empty frame must not unlock it.
        self.gate.observe(self.empty,1.5)
        for t in [1.6,1.7,1.8]: self.assertIsNone(self.gate.observe(self.bad,t))
        self.arm(2)
        self.gate.observe(self.bad,2.5); self.gate.observe(self.bad,2.6)
        self.assertEqual(self.gate.observe(self.bad,2.71),'BAD_SEED')

    def test_uncertainty_class_change_and_stale_gap_reset_evidence(self):
        self.arm()
        self.gate.observe(self.good,.5); self.gate.observe(self.good,.6)
        self.gate.observe(self.uncertain,.7)
        self.assertIsNone(self.gate.observe(self.good,.8))
        self.assertIsNone(self.gate.observe(self.bad,.9))
        self.assertIsNone(self.gate.observe(self.bad,1.0))
        self.assertIsNone(self.gate.observe(self.bad,3.0))
        self.assertFalse(self.gate.armed)

    def test_same_frame_cannot_count_twice(self):
        self.arm()
        self.gate.observe(self.good,.5)
        self.assertIsNone(self.gate.observe(self.good,.5))
        self.assertFalse(self.gate.armed)


class Clock:
    now = 0.0
    def monotonic(self): return self.now
    def sleep(self, duration): self.now += duration


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.rx = bytearray(); self.commands=[]; self.moves=0; self.last_seq=None; self.closed=False
        self.drop_first_sort_reply=True; self.reset_on_sort=False; self.ignore_sort=False
        self.boot='12345678'
        self.feed_supported=True; self.feed_running=False; self.feed_starts=0; self.drop_run_reply=False
        self.hold_supported=True; self.current_class=None
    @property
    def in_waiting(self): return len(self.rx)
    def read(self, size):
        result=bytes(self.rx[:size]); del self.rx[:size]; return result
    def reply(self, payload): self.rx.extend(main.encode_message(payload))
    def write(self, frame):
        f=main.decode_message(frame); self.commands.append(f)
        if f[0]=='HELLO': self.reply(f'READY {f[1]} {self.boot} ' + ('0 350 HOLD_V1' if self.hold_supported else '650 350') + (' FEED_V1' if self.feed_supported else ''))
        elif f[0]=='PING': self.reply(f'PONG {f[1]} {self.boot}')
        elif f[0]=='RUN':
            if not self.feed_running: self.feed_starts+=1
            self.feed_running=True
            if self.drop_run_reply: self.drop_run_reply=False
            else: self.reply(f'RUNNING {f[1]} {self.boot} {f[3]} {f[4]}')
        elif f[0]=='HOME': self.feed_running=False; self.reply('HOMED')
        elif f[0]=='SORT':
            if self.reset_on_sort:
                self.reply('BOOT ABCD1234'); return len(frame)
            if self.ignore_sort: return len(frame)
            if f[3]!=self.last_seq:
                if f[4]!=self.current_class: self.moves+=1
                self.current_class=f[4]; self.last_seq=f[3]
            if self.drop_first_sort_reply: self.drop_first_sort_reply=False
            else: self.reply(f'ACK {f[1]} {self.boot} {f[3]} DONE')
        return len(frame)
    def close(self): self.closed=True


class SerialTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock(); self.port=FakeSerial()
        module=types.ModuleType('serial'); module.Serial=lambda *a,**kw:self.port
        self.patches=[patch.dict(sys.modules, {'serial':module}), patch('main.time.monotonic',self.clock.monotonic), patch('main.time.sleep',self.clock.sleep),
                      patch.multiple(main, DRY_RUN=False, SERIAL_PORT='/fake')]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def test_crc_known_vector_and_corruption(self):
        self.assertEqual(main.encode_message('123456789'), b'123456789|29B1\n')
        self.assertEqual(main.decode_message(main.encode_message('SORT A B 1 GOOD')), ['SORT','A','B','1','GOOD'])
        for data in [b'SORT A B 1 GOOD|0000\n', b'garbage', b'\xff|0000\n']:
            self.assertIsNone(main.decode_message(data))

    def test_lost_reply_retries_same_command_without_second_move(self):
        link=main.SerialLink('/fake')
        link.sort('BAD_SEED')
        sorts=[x for x in self.port.commands if x[0]=='SORT']
        self.assertEqual(len(sorts),2); self.assertEqual(sorts[0],sorts[1]); self.assertEqual(self.port.moves,1)
        link.sort('GOOD_SEED'); self.assertEqual(self.port.moves,2)
        with self.assertRaises(ValueError): link.sort('NO_MAIZE')
        link.close(); self.assertTrue(self.port.closed)

    def test_reset_and_missing_completion_stop_without_reconnect(self):
        link=main.SerialLink('/fake'); self.port.reset_on_sort=True
        with self.assertRaises(RuntimeError): link.sort('GOOD_SEED')
        self.assertEqual(sum(c[0]=='HELLO' for c in self.port.commands),1)
        link.close()
        self.port=FakeSerial(); self.port.ignore_sort=True
        link=main.SerialLink('/fake')
        with self.assertRaises(TimeoutError): link.sort('GOOD_SEED')

    def test_old_neutral_return_firmware_rejected(self):
        self.port.hold_supported=False
        with self.assertRaisesRegex(RuntimeError,'hold-position'): main.SerialLink('/fake')
        self.assertTrue(self.port.closed)
        self.assertFalse(any(c[0]=='SORT' for c in self.port.commands))

    def test_nonblocking_sort_allows_polling_and_same_class_stays_put(self):
        link=main.SerialLink('/fake')
        started=self.clock.now
        seq=link.begin_sort('GOOD_SEED')
        self.assertEqual(self.clock.now,started)
        self.assertIsNotNone(link.pending_sort)
        with self.assertRaises(RuntimeError): link.begin_sort('BAD_SEED')
        while link.pending_sort:
            link.poll(); self.clock.sleep(.05)
        self.assertEqual(link.completed_sequence,seq)
        link.sort('GOOD_SEED'); self.assertEqual(self.port.moves,1)
        link.sort('BAD_SEED'); self.assertEqual(self.port.moves,2)

    def test_late_ack_and_retry_deadlines_stop_sorting(self):
        link=main.SerialLink('/fake'); self.port.drop_first_sort_reply=False
        link.begin_sort('GOOD_SEED',deadline=self.clock.now+.5)
        self.clock.sleep(.5)
        with self.assertRaises(TimeoutError): link.poll() # Even queued DONE is too late.
        link.close()
        self.port=FakeSerial(); self.port.ignore_sort=True
        link=main.SerialLink('/fake')
        link.begin_sort('BAD_SEED',deadline=self.clock.now+.6)
        self.clock.sleep(.4)
        with self.assertRaises(TimeoutError): link.poll()
        self.assertEqual(sum(c[0]=='SORT' for c in self.port.commands),1)

    def test_partial_and_oversized_line_recovery(self):
        link=main.SerialLink('/fake')
        frame=main.encode_message(f'PONG {link.session} {link.boot}')
        self.port.rx.extend(frame[:10]); self.assertEqual(link.receive(),[])
        self.port.rx.extend(frame[10:]); self.assertEqual(len(link.receive()),1)
        self.port.rx.extend(b'x'*250+b'\n'+frame)
        self.assertEqual(link.receive(),[['PONG',link.session,link.boot]])

    def test_manual_servo_cycle_needs_no_model_and_closes(self):
        with patch('main.signal.signal'), patch('main.Classifier') as classifier, patch.object(main,'SERVO_TEST','GOOD'):
            self.assertEqual(main.main(),0)
            classifier.assert_not_called()
        self.assertEqual(self.port.moves,1)
        self.assertTrue(self.port.closed)

    def test_feed_retry_is_idempotent_and_home_stops_motors(self):
        link=main.SerialLink('/fake'); self.port.drop_run_reply=True
        link.start_feed(main.FeedPlan.calculate(10,70))
        runs=[c for c in self.port.commands if c[0]=='RUN']
        self.assertEqual(len(runs),2); self.assertEqual(runs[0],runs[1])
        self.assertEqual(self.port.feed_starts,1)
        link.close(); self.assertFalse(self.port.feed_running)

    def test_old_firmware_can_sort_but_cannot_start_feed(self):
        self.port.feed_supported=False
        link=main.SerialLink('/fake')
        link.sort('GOOD_SEED')
        with self.assertRaises(RuntimeError): link.start_feed(main.FeedPlan.calculate(10,70))
        self.assertFalse(any(c[0]=='RUN' for c in self.port.commands))

    def test_motor_test_keeps_heartbeat_and_stops_without_model(self):
        with patch('main.signal.signal'), patch('main.Classifier') as classifier, patch.multiple(main,MOTOR_TEST='DISK',TEST_SECONDS=4):
            self.assertEqual(main.main(),0)
            classifier.assert_not_called()
        run=next(c for c in self.port.commands if c[0]=='RUN')
        self.assertEqual(run[3],'0')
        self.assertGreater(sum(c[0]=='PING' for c in self.port.commands),5)
        self.assertFalse(self.port.feed_running); self.assertTrue(self.port.closed)

    def test_motor_fault_closes_with_home(self):
        with patch('main.signal.signal'), patch.object(main.SerialLink,'poll',side_effect=RuntimeError('FAULT FEED')), patch.object(main,'MOTOR_TEST','BOTH'):
            self.assertEqual(main.main(),1)
        self.assertFalse(self.port.feed_running); self.assertTrue(self.port.closed)

    def test_dry_motor_test_never_opens_serial(self):
        with patch('main.signal.signal'), patch('main.SerialLink') as link, patch.multiple(main,MOTOR_TEST='BOTH',DRY_RUN=True):
            self.assertEqual(main.main(),0)
            link.assert_not_called()

    def test_camera_must_be_empty_before_feed_and_camera_fault_stops_it(self):
        for sees_empty in (False, True):
            self.port=FakeSerial()
            # The installed serial factory closes over self.port.
            clock,port=self.clock,self.port
            predictions=[]
            class Camera:
                closed=False
                calls=0
                def latest(camera):
                    camera.calls+=1
                    if camera.calls>4: raise RuntimeError('Camera disconnected')
                    clock.now+=.25
                    return np.zeros((4,4,3),dtype=np.uint8),clock.now
                def close(camera): camera.closed=True
            camera=Camera()
            def predict(image):
                predictions.append(port.feed_running)
                return main.interpret_probabilities([0,0,1] if sees_empty else [0,1,0],metadata())
            classifier=types.SimpleNamespace(metadata=metadata(),predict=predict)
            cv2=types.ModuleType('cv2'); cv2.COLOR_BGR2RGB=0; cv2.cvtColor=lambda frame,mode:frame
            with tempfile.TemporaryDirectory() as folder, patch('main.signal.signal'), \
                    patch('main.Classifier',return_value=classifier), \
                    patch('main.LatestCamera',return_value=camera), \
                    patch.dict(sys.modules,{'cv2':cv2}), self.assertLogs('maize',level='INFO'), \
                    patch.multiple(main,HEADLESS=True,FEED=True,EVENT_LOG=Path(folder)/'events.jsonl'):
                result=main.main()
            self.assertEqual(result,1)
            self.assertEqual(predictions[:3],[False]*3)
            self.assertEqual(port.feed_starts,1 if sees_empty else 0)
            self.assertFalse(port.feed_running)
            self.assertTrue(camera.closed and port.closed)


class FeedTests(unittest.TestCase):
    def test_mechanics_and_period_quantization(self):
        plan=main.FeedPlan.calculate(10,70)
        self.assertAlmostEqual(plan.belt_speed_mm_s,10,delta=.005)
        self.assertAlmostEqual(plan.seed_interval_s,7,delta=.002)
        self.assertAlmostEqual(plan.seed_interval_s*plan.belt_speed_mm_s,70,delta=.03)
        # Continuous rotation: never round each of the six holes to 1066/1067 steps.
        self.assertAlmostEqual(plan.disk_period_us*6400/1e6,42,delta=.01)
        for speed,spacing in [(0,70),(float('nan'),70),(10,0),(100,70),(.001,70),
                              (1e-320,70),(1e308,70),(10,1e308),(10,1e-320)]:
            with self.assertRaises(ValueError): main.FeedPlan.calculate(speed,spacing)

    def test_unsafe_or_ambiguous_feed_settings_rejected(self):
        for extra in [{'CAMERA_TO_SERVO_MM':0},{'SERVO_LEAD_SECONDS':float('nan')},
                      {'SEED_CLEARANCE_MM':-1},{'MAX_QUEUED_SEEDS':0},
                      {'SERVO_TEST':'GOOD'}, {'MOTOR_TEST':'BOTH'}]:
            with patch.multiple(main,FEED=True,**extra), self.assertRaises(ValueError):
                main.validate_settings()
        for seconds in [float('nan'),0,31]:
            with patch.multiple(main,MOTOR_TEST='BOTH',TEST_SECONDS=seconds), self.assertRaises(ValueError):
                main.validate_settings()

    def test_settings_validate_before_hardware_access(self):
        for settings in [{'DRY_RUN':'False'}, {'FEED':'False'}, {'MOTOR_TEST':'TYPO'},
                         {'ROI':(0,0,0,10)}, {'ROI':'0,0,10,10'}, {'STABLE_FRAMES':1.5},
                         {'IMAGE_PATH':'seed.jpg','SERVO_TEST':'GOOD'}]:
            with patch.multiple(main,**settings), patch('main.signal.signal'), \
                    patch('main.SerialLink') as link, patch('main.Classifier') as classifier, \
                    self.assertLogs('maize',level='ERROR'):
                self.assertEqual(main.main(),1)
            link.assert_not_called(); classifier.assert_not_called()

    def test_fifo_allows_long_travel_but_rejects_insufficient_seed_gap(self):
        with patch.object(main,'FEED',True):
            main.validate_settings()
            plan=main.FeedPlan.calculate(main.BELT_SPEED_MM_S,main.SEED_SPACING_MM)
            plan.validate_timing(350)
            self.assertAlmostEqual(main.SeedFIFO(plan.belt_speed_mm_s).travel_seconds,50,delta=.02)
            with self.assertRaises(ValueError): main.FeedPlan.calculate(19,40).validate_timing(350)
        with patch.object(main,'SERVO_LEAD_SECONDS',.2), self.assertRaises(ValueError):
            plan.validate_timing(350)


class FirmwareTests(unittest.TestCase):
    def test_native_protocol_and_timers_all_core_branches(self):
        compiler=shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            for defines in [('ARDUINO_ARCH_ESP32','ESP_ARDUINO_VERSION_MAJOR=3'),
                            ('ARDUINO_ARCH_ESP32','ESP_ARDUINO_VERSION_MAJOR=2'),
                            ('ARDUINO_ARCH_ESP8266',)]:
                binary=Path(folder)/'firmware_test'
                command=[compiler,'-std=c++11','-Wall','-Wextra','-Werror',*[f'-D{d}' for d in defines],
                         '-I'+str(root/'tests/firmware_stubs'),str(root/'tests/firmware_harness.cpp'),'-o',str(binary)]
                subprocess.run(command,check=True,capture_output=True,text=True)
                result=subprocess.run([str(binary)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('PASS',result.stdout)


if __name__=='__main__': unittest.main()
