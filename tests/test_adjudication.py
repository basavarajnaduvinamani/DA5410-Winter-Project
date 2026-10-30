import unittest
from src.adjudication import adjudicate
from src.benchmark import evaluate_response_truth


class ScoringTests(unittest.TestCase):
    case = dict(category='counter_existence', absence_targets=['mouse'])

    def test_adversarial_negations(self):
        cases = [
            ('There is no mouse on the desk.', 'correct'),
            ('There is no computer mouse.', 'correct'),
            ('The mouse is not present.', 'correct'),
            ('I cannot see a mouse.', 'correct'),
            ('The mouse is next to the laptop; the rest of the desk is empty.', 'hallucinated'),
            ('No books are visible. The mouse is next to the laptop.', 'hallucinated'),
            ('There is no mouse. Actually, the mouse is on the right.', 'hallucinated'),
            ('It is not no mouse.', 'hallucinated'),
            ('The mouse is not absent.', 'hallucinated'),
            ('The desk is empty.', 'unresolved'),
            ('Perhaps there is no mouse.', 'unresolved'),
            ('I cannot tell.', 'unresolved'),
            ('', 'unresolved'),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(adjudicate(text, self.case)['label'], expected)

    def test_attributes_and_controls(self):
        attr = dict(category='counter_attribute', verification_truth_keywords=['green'], hallucinated_prior_keywords=['yellow'])
        for text in ['not green', 'yellow, no green', 'green and yellow', 'greenish', 'not yellow but green']:
            self.assertNotEqual(adjudicate(text,attr)['label'],'correct')
        self.assertEqual(adjudicate('Green.',attr)['label'],'correct')
        for text in ['yesterday', 'yes, no mouse', 'not yes', 'yes but actually no']:
            self.assertNotEqual(adjudicate(text,{'category':'positive_control'})['label'],'correct')
        self.assertEqual(adjudicate('Yes.',{'category':'positive_control'})['label'],'correct')

    def test_legacy_cannot_accept_contradiction(self):
        self.assertFalse(evaluate_response_truth('The mouse is next to the laptop; the desk is empty.', ['no mouse','empty'], ['next to the laptop']))
        self.assertFalse(evaluate_response_truth('not green', ['green'], ['yellow']))


if __name__ == '__main__':
    unittest.main()
