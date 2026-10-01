"""Execute the notebook's actual data/calibration cells without requiring a GPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import main

NOTEBOOK = Path(__file__).resolve().parents[1]/'Maize_Seed_Classification_Model.ipynb'


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook=json.loads(NOTEBOOK.read_text())

    def code(self,index): return ''.join(self.notebook['cells'][index]['source'])

    def test_all_cells_parse(self):
        for index,cell in enumerate(self.notebook['cells']):
            if cell['cell_type']=='code':
                source='\n'.join(line for line in self.code(index).splitlines() if not line.startswith(('!','%')))
                ast.parse(source,filename=f'cell_{index}')

    def test_duplicate_filter_groups_augmentation_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory); output=root/'outputs'; output.mkdir()
            group_rows=[]
            for label,name in enumerate(main.CLASSES):
                folder=root/name; folder.mkdir()
                for index in range(40):
                    path=folder/f'{index:03d}.png'
                    pixels=np.random.default_rng(label*100+index).integers(0,256,(24,32,3),dtype=np.uint8)
                    Image.fromarray(pixels).save(path)
                    group_rows.append({'relative_path':f'{name}/{path.name}', 'group_id':f'session_{index//4}'})
            duplicate=root/'BAD_SEED'/'duplicate.png'
            duplicate.write_bytes((root/'BAD_SEED'/'000.png').read_bytes())
            (root/'GOOD_SEED'/'broken.jpg').write_bytes(b'not an image')
            pd.DataFrame(group_rows).to_csv(root/'source_groups.csv', index=False)
            env=dict(Path=Path, Image=Image, np=np, pd=pd, json=json,
                     DATA_ROOT=root, OUTPUT_DIR=output, IMAGE_SIZE=(32,32), SEED=42,
                     CLASS_NAMES=main.CLASSES, CLASS_TO_INDEX=dict(zip(main.CLASSES,range(3))),
                     VAL_FRACTION=.15, TEST_FRACTION=.15, GROUPS_CSV=root/'source_groups.csv',
                     GroupShuffleSplit=GroupShuffleSplit, train_test_split=train_test_split,
                     tqdm=lambda values,**kwargs:values, AUGMENTED_ROOT=root/'augmented', TARGET_TRAIN_IMAGES_PER_CLASS=35)
            exec(self.code(10),env)
            self.assertEqual(len(env['df']),120)
            self.assertEqual(len(env['duplicates']),1); self.assertEqual(len(env['corrupt_files']),1)
            exec(self.code(11),env)
            heldout=env['val_df'].copy(),env['test_df'].copy()
            exec(self.code(13),env)
            self.assertEqual(env['train_df'].groupby('label_name').size().to_dict(),dict.fromkeys(main.CLASSES,35))
            self.assertEqual(len(list(env['AUGMENTED_TRAIN_DIR'].rglob('*.jpg'))),105)
            pd.testing.assert_frame_equal(heldout[0],env['val_df']); pd.testing.assert_frame_equal(heldout[1],env['test_df'])
            sources=set(env['train_df'].source_path)
            self.assertTrue(sources.isdisjoint(set(env['val_df'].path)|set(env['test_df'].path)))
            before=env['train_df'].copy()
            exec(self.code(13),env)
            pd.testing.assert_frame_equal(before,env['train_df'])
            self.assertEqual(len(list(env['AUGMENTED_TRAIN_DIR'].rglob('*.jpg'))),105)
            # A conflicting label is a hard data error, not an arbitrary kept label.
            (root/'GOOD_SEED'/'conflicting.png').write_bytes((root/'BAD_SEED'/'000.png').read_bytes())
            with self.assertRaisesRegex(ValueError,'conflicting labels'): exec(self.code(10),env)

    def test_calibration_matches_runtime_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            labels=np.repeat([0,1,2],10)
            probabilities=np.full((30,3),.02); probabilities[np.arange(30),labels]=.96
            env=dict(np=np,pd=pd,json=json,model=types.SimpleNamespace(predict=lambda *a,**kw:probabilities),
                     val_ds=None,val_df=pd.DataFrame({'label':labels}),no_idx=2,
                     CLASS_NAMES=main.CLASSES,CLASS_TO_INDEX=dict(zip(main.CLASSES,range(3))), OUTPUT_DIR=Path(directory))
            exec(self.code(27),env)
            self.assertTrue(env['ACTUATION_ENABLED'])
            accepted=env['accepted_mask'](probabilities)
            runtime=np.array([main.interpret_probabilities(row,{'decision_policy':env['DECISION_POLICY']}).eligible for row in probabilities])
            np.testing.assert_array_equal(accepted,runtime)
            self.assertEqual(int(accepted.sum()),20)
            # Float32 softmax outputs near a decimal threshold must make the same decision on the Pi.
            for dtype in [np.float32, np.float64]:
                boundary=probabilities.astype(dtype)
                runtime=np.array([main.interpret_probabilities(row,{'decision_policy':env['DECISION_POLICY']}).eligible for row in boundary])
                np.testing.assert_array_equal(env['accepted_mask'](boundary),runtime)
            # BAD predictions indistinguishable from hard negatives cannot get an acceptable policy.
            probabilities[labels==2]=[.96,.02,.02]
            exec(self.code(27),env)
            self.assertFalse(env['ACTUATION_ENABLED'])
            self.assertFalse(env['accepted_mask'](probabilities).any())

    def test_notebook_and_pi_preprocessing_identical(self):
        tree=ast.parse(self.code(31))
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_rgb')
        env=dict(np=np,Image=Image,ImageOps=ImageOps,IMAGE_SIZE=(17,23))
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook_preprocessing>','exec'),env)
        image=Image.fromarray(np.random.default_rng(1).integers(0,256,(41,39,3),dtype=np.uint8))
        np.testing.assert_array_equal(main.prepare_rgb(image,(17,23)),env['prepare_rgb'](image))


if __name__=='__main__': unittest.main()
