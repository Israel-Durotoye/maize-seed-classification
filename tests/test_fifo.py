import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import main
from test_main import Clock, FakeSerial, metadata


class FIFOTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.clock.now = 0
        self.port = FakeSerial()
        module = types.ModuleType('serial')
        module.Serial = lambda *a, **kw: self.port
        for p in [patch.dict('sys.modules', {'serial': module}),
                  patch('main.time.monotonic', self.clock.monotonic),
                  patch('main.time.sleep', self.clock.sleep),
                  patch.multiple(main, CAMERA_TO_SERVO_MM=500, SERVO_LEAD_SECONDS=2,
                                 SEED_CLEARANCE_MM=10, MAX_QUEUED_SEEDS=64)]:
            p.start(); self.addCleanup(p.stop)
        self.fifo = main.SeedFIFO(10)

    def drive_until(self, stop, link):
        events = []
        while self.clock.now < stop:
            events.extend(self.fifo.service(link))
            self.clock.sleep(.05)
        return events

    def test_50cm_travel_and_multiple_seeds_in_flight(self):
        for i, label in enumerate(['GOOD_SEED', 'GOOD_SEED', 'BAD_SEED', 'GOOD_SEED']):
            self.clock.now = i * 7
            seed = self.fifo.enqueue(label, self.clock.now)
            self.assertEqual(seed.arrival_at, self.clock.now + 50)
            self.assertEqual(seed.command_at, self.clock.now + 48)
        self.assertEqual(len(self.fifo.queue), 4)
        link = main.SerialLink('/fake')
        events = self.drive_until(47.9, link)
        self.assertEqual(self.port.moves, 0)
        events += self.drive_until(74, link)
        sorts = [x for x in self.port.commands if x[0] == 'SORT']
        unique = {x[3]: x[4] for x in sorts}
        self.assertEqual(list(unique.values()), ['GOOD', 'GOOD', 'BAD', 'GOOD'])
        self.assertEqual(self.port.moves, 3)
        self.assertEqual([s.seed_id for event, s in events if event == 'seed_passed_estimate'], [1,2,3,4])
        self.assertFalse(self.fifo.queue)

    def test_queue_keeps_head_until_it_physically_clears(self):
        seed = self.fifo.enqueue('GOOD_SEED', 0)
        self.clock.now = 48
        self.fifo.service(None)
        self.assertTrue(seed.ready)
        self.clock.now = 50.99
        self.assertEqual(self.fifo.service(None), [])
        self.assertEqual(len(self.fifo.queue), 1)
        self.clock.now = 51
        self.assertEqual(self.fifo.service(None)[0][0], 'seed_passed_estimate')
        self.assertFalse(self.fifo.queue)

    def test_rejects_overlapping_seed_clearance_and_out_of_order_events(self):
        self.fifo.enqueue('GOOD_SEED', 0)
        for timestamp in [0, -1, 2.9, 3.0]:
            with self.assertRaises(RuntimeError): self.fifo.enqueue('BAD_SEED', timestamp)

    def test_missed_deadline_does_not_flush_late_commands(self):
        self.fifo.enqueue('GOOD_SEED', 0)
        self.fifo.enqueue('BAD_SEED', 7)
        self.clock.now = 49.8
        link = main.SerialLink('/fake')
        with self.assertRaises(TimeoutError): self.fifo.service(link)
        self.assertFalse(any(c[0] == 'SORT' for c in self.port.commands))

    def test_no_done_ack_by_arrival_stops(self):
        self.fifo.enqueue('GOOD_SEED', 0)
        self.port.ignore_sort = True
        link = main.SerialLink('/fake')
        with self.assertRaises(TimeoutError): self.drive_until(51, link)
        self.assertEqual(len(self.fifo.queue), 1)

    def test_overflow_and_late_classification_stop(self):
        with patch.object(main, 'MAX_QUEUED_SEEDS', 1):
            self.fifo.enqueue('GOOD_SEED', 0)
            with self.assertRaises(RuntimeError): self.fifo.enqueue('BAD_SEED', 7)
        self.clock.now = 100
        with self.assertRaises(TimeoutError): main.SeedFIFO(10).enqueue('GOOD_SEED', 0)

    def test_short_distance_or_servo_lead_is_rejected(self):
        with patch.object(main, 'CAMERA_TO_SERVO_MM', 10), self.assertRaises(ValueError):
            main.SeedFIFO(10)
        with patch.object(main, 'SERVO_LEAD_SECONDS', .5), self.assertRaises(ValueError):
            main.SeedFIFO(10)

    def test_gate_timestamps_first_sighting_and_rejects_uncertain_seed_once(self):
        gate = main.DecisionGate()
        empty = main.interpret_probabilities([0,0,1], metadata())
        uncertain = main.interpret_probabilities([.45,.5,.05], metadata())
        good = main.interpret_probabilities([0,1,0], metadata())
        for t in [0,.25,.5]: gate.observe(empty, t)
        gate.observe(uncertain, .75)
        gate.observe(good, 1)
        gate.observe(good, 1.25)
        self.assertEqual(gate.observe(good, 1.5), 'GOOD_SEED')
        self.assertEqual(gate.event_seen_at, .75)
        for t in [1.75,2,2.25,2.5]: gate.observe(empty, t)
        gate.observe(uncertain, 2.75)
        gate.observe(uncertain, 3)
        gate.observe(empty, 3.25); gate.observe(empty, 3.5)
        self.assertEqual(gate.observe(empty, 3.75), 'BAD_SEED')
        self.assertEqual(gate.event_seen_at, 2.75)
        self.assertEqual(gate.event_reason, 'uncertain_reject')
        self.assertIsNone(gate.reject_pending(4))

    def test_camera_keeps_classifying_during_travel_and_servo_commands(self):
        clock, port = self.clock, self.port
        class Camera:
            closed = False
            def latest(camera):
                clock.now += .25
                if clock.now > 80: raise RuntimeError('End simulated camera stream')
                return np.zeros((4,4,3), dtype=np.uint8), clock.now
            def close(camera): camera.closed = True
        camera = Camera()
        classified_after_first_command = []
        def predict(image):
            if any(c[0] == 'SORT' for c in port.commands):
                classified_after_first_command.append(clock.now)
            for start, label in [(1,'GOOD_SEED'), (8,'GOOD_SEED'), (15,'BAD_SEED'), (22,'GOOD_SEED')]:
                if start <= clock.now <= start+1.5:
                    return main.interpret_probabilities([1,0,0] if label == 'BAD_SEED' else [0,1,0], metadata())
            return main.interpret_probabilities([0,0,1], metadata())
        classifier = types.SimpleNamespace(metadata=metadata(), predict=predict)
        cv2 = types.ModuleType('cv2'); cv2.COLOR_BGR2RGB = 0
        cv2.cvtColor = lambda frame, mode: frame
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'events.jsonl'
            with patch.multiple(main, DRY_RUN=False, FEED=True, HEADLESS=True, EVENT_LOG=log), \
                    patch('main.signal.signal'), patch('main.Classifier', return_value=classifier), \
                    patch('main.LatestCamera', return_value=camera), patch.dict('sys.modules', {'cv2': cv2}), \
                    self.assertLogs('maize', level='INFO'):
                self.assertEqual(main.main(), 1) # Deliberately end via camera failure.
            events = [json.loads(line) for line in log.read_text().splitlines()]
        queued = [e for e in events if e['event'] == 'seed_queued']
        self.assertEqual([e['class'] for e in queued], ['GOOD_SEED','GOOD_SEED','BAD_SEED','GOOD_SEED'])
        self.assertEqual([e['seen_at'] for e in queued], [1,8,15,22])
        self.assertEqual(sum(e['event'] == 'seed_passed_estimate' for e in events), 4)
        self.assertGreater(len(classified_after_first_command), 50)
        self.assertEqual(port.moves, 3)
        self.assertTrue(camera.closed and port.closed)
        self.assertFalse(port.feed_running)


if __name__ == '__main__':
    unittest.main()
