"""Check that paired controls and utility losses keep the correct question alignment."""
import unittest
from scripts.make_report import Run

class PairedReport(unittest.TestCase):
    def test_control_pairing_and_recall_loss(self):
        run=Run.__new__(Run);run.seed=7;run.iterations=50;run.rows={}
        for method in ['baseline','slerp_pca','add_mean_difference','random_direction','shuffled_labels']:
            run.rows[method]={}
            for image in range(4):
                for label in ['yes','no']:
                    uid=f'{image}:{label}'
                    pred='no' if method in ['slerp_pca','shuffled_labels'] else 'yes'
                    run.rows[method][uid]=dict(uid=uid,image=str(image),partition='test',pope_split='random',label=label,pred=pred,strength=.05)
            run.rows[method]=dict(reversed(list(run.rows[method].items())))
        sl=run.steering()['slerp_pca']
        self.assertEqual(sl['delta_recall']['estimate'],-1)
        self.assertEqual(sl['delta_false_yes_rate']['estimate'],-1)
        self.assertEqual(sl['versus_controls_false_yes']['random_direction']['ci95'],[-1,-1])
        self.assertEqual(sl['versus_controls_false_yes']['shuffled_labels']['ci95'],[0,0])
if __name__=='__main__':unittest.main()
