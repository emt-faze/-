import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from accuracy_eval import evaluate, normalize, distance, prepare, split_digest, REQUIRED_TAGS
from ocr_core import OCR_PROFILE


class AccuracyTests(unittest.TestCase):
    def dataset(self, root):
        image=root/'original.png'
        image.write_bytes(b'unchanged original pixel fixture')
        samples=[]
        for index in range(50):
            text=f'不要将第{index:02d}号替换。这是一条用于检验计分方法的台词。'
            samples.append(dict(dialogue_id=str(index),body_image=str(image),
                body_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                split='tune' if index<20 else 'acceptance',ocr_profile=OCR_PROFILE,
                original_text=text,review_text=text,displayed_text=text,
                truth_body=text,truth_speaker='',speaker='',review_speaker='',
                truth_checked=True,label_source='synthetic_test_only',tags=sorted(REQUIRED_TAGS),status='verified'))
        manifest=dict(samples=samples,split_frozen=True,ocr_profile=OCR_PROFILE,
                      split_sha256=split_digest(samples))
        path=root/'dataset.json'
        path.write_text(json.dumps(manifest),encoding='utf-8')
        return path,manifest

    def test_character_metric_keeps_negation_digits_punctuation_and_wrapping_only(self):
        self.assertEqual(normalize('不要\n开门！'),'不要开门！')
        self.assertEqual(distance('不要开门！','要开门！'),1)
        self.assertEqual(distance('10','11'),1)
        self.assertEqual(distance('对。','对'),1)

    def test_insufficient_or_unlabelled_real_samples_never_claim_score(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            root=Path(directory)
            path=root/'dataset.json'
            manifest=prepare(root/'records',path)
            self.assertEqual(manifest['samples'],[])
            result=evaluate(path)
            self.assertEqual(result['status'],'not_measured')
            self.assertNotIn('displayed',result)
            path,manifest=self.dataset(root)
            manifest['samples'][30]['truth_checked']=False
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'not_measured')

    def test_29_of_30_passes_but_28_fails_exact_sentence_target(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            root=Path(directory)
            path,manifest=self.dataset(root)
            for index,expected in ((20,'passed'),(21,'failed')):
                manifest['samples'][index]['displayed_text']+='错'
                path.write_text(json.dumps(manifest),encoding='utf-8')
                result=evaluate(path)
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['displayed']['exact_sentences'],49-index)
                self.assertLess(result['displayed']['cer'],.01)

    def test_changed_split_image_and_profile_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            root=Path(directory)
            path,manifest=self.dataset(root)
            manifest['samples'][0]['split']='acceptance'
            manifest['samples'][20]['split']='tune'
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'not_measured')
            path,manifest=self.dataset(root)
            (root/'original.png').write_bytes(b'changed pixels')
            self.assertEqual(evaluate(path)['status'],'not_measured')
            path,manifest=self.dataset(root)
            manifest['samples'][0]['ocr_profile']='other'
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'not_measured')

    def test_missing_required_coverage_does_not_pass_acceptance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            path,manifest=self.dataset(Path(directory))
            for sample in manifest['samples']:
                sample['tags']=['short']
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'incomplete_coverage')

    def test_suspected_clipping_requires_image_confirmation_instead_of_silent_exclusion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'validation') as directory:
            root=Path(directory)
            path,manifest=self.dataset(root)
            manifest['samples'][20]['source_clipped']=True
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'not_measured')
            manifest['samples'][20]['source_complete_confirmed']=True
            path.write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(evaluate(path)['status'],'passed')

    def test_seeded_tuning_stays_fixed_and_seen_dialogues_cannot_fill_holdout(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'validation') as directory:
            root=Path(directory)
            path,manifest=self.dataset(root)
            manifest['samples']=manifest['samples'][:20]
            manifest.update(split_frozen=False,split_seed=20261029,known_tuning_texts=[],tuning_frozen_ids=[str(i) for i in range(20)],
                            development_exclusions=['已经用于调试的台词'])
            path.write_text(json.dumps(manifest),encoding='utf-8')
            records=root/'records'; records.mkdir()
            for index in range(31):
                text='已经用于调试的台词' if index==0 else f'全新剧情台词{index}'
                record=dict(source_kind='live_game_capture',ocr_profile=OCR_PROFILE,dialogue_id=f'new-{index}',
                            body_image=str(root/'original.png'),body=text,original_text=text,review_text=text,status='verified')
                (records/f'{index:02d}.json').write_text(json.dumps(record),encoding='utf-8')
            result=prepare(records,path)
            self.assertEqual({s['dialogue_id'] for s in result['samples'] if s['split']=='tune'},set(map(str,range(20))))
            self.assertEqual(sum(s['split']=='acceptance' for s in result['samples']),30)
            self.assertNotIn('new-0',{s['dialogue_id'] for s in result['samples']})

    def test_development_example_stays_in_tuning_and_truth_is_not_prefilled(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            root=Path(directory)
            records=root/'records'
            records.mkdir()
            image=records/'source.png'
            image.write_bytes(b'test fixture')
            for index in range(50):
                text='……味道如何？' if index==0 else f'不同的句子{index}'
                record=dict(source_kind='live_game_capture',ocr_profile=OCR_PROFILE,dialogue_id=str(index),
                            body_image='source.png',body=text,original_text=text,review_text=text,status='verified')
                (records/f'{index:02d}.json').write_text(json.dumps(record),encoding='utf-8')
            path=root/'dataset.json'
            result=prepare(records,path)
            self.assertEqual(result['samples'][0]['split'],'tune')
            self.assertEqual(sum(s['split']=='acceptance' for s in result['samples']),30)
            self.assertTrue(all(s['truth_body'] is None for s in result['samples']))
            self.assertEqual(prepare(records,path)['split_sha256'],result['split_sha256'])

if __name__=='__main__':
    unittest.main()

