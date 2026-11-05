"""Round-2 review of the 34 rows excluded by the 2026-10-30 pixel audit.

Writes dataset/exclusion_review_2026-10-30.csv and a PROPOSED relabelled
manifest. Relabelling never edits a prompt or image: a row is "fixable" only
when its unchanged prompt presupposes an object that is plainly absent, so the
row becomes a counter_existence (false-premise) trap with explicit absence
targets. Judgments come from one model-assisted visual pass over the images
and need an independent human check before dataset/counterprior_corrected.json
is replaced; the default runner keeps using the 16-row manifest.
"""
import csv
import json
from collections import Counter
from pathlib import Path

REVIEWER = 'Claude visual pass over dataset/images, 2026-10-31; independent human confirmation pending'

# id -> (what the image shows, fixable, absence targets or reason not fixable)
JUDGMENTS = {
    'attr_02_blue_strawberry': ('pile of red strawberries, no plate', False, 'object present with its typical colour; relabel would remove the counter-prior'),
    'attr_03_purple_carrot': ('orange carrots on a dark surface', False, 'object present with typical colour'),
    'attr_04_square_orange': ('many round oranges', False, 'object present with typical shape'),
    'attr_05_pink_lemon': ('plate of toast, fried egg, tomato and lettuce; no lemon', True, ['lemon']),
    'attr_06_transparent_mug': ('paper takeaway coffee cups', False, 'cup present but not a mug; answer depends on judgment of cup vs mug'),
    'attr_07_white_crow': ('orange-yellow parrot on a branch', False, 'a bird is present; whether "the crow" is absent is a species judgment'),
    'attr_08_yellow_traffic_light': ('two hands holding a plant sprig over grass; no traffic light', True, ['traffic light']),
    'attr_09_black_apple': ('hand holding a red apple', False, 'object present with typical colour'),
    'attr_10_golden_beet': ('mixed salad bowl', False, 'root vegetable may be present in the salad; not decidable at this resolution'),
    'attr_11_white_eggplant': ('red drink in a jar with lemon slices; no eggplant', True, ['eggplant']),
    'attr_12_blue_corn': ('yellow corn cobs', False, 'object present with typical colour'),
    'attr_13_green_rose': ('castle on a hill above fog; no rose (same pixels as abs_10)', True, ['rose', 'petals']),
    'attr_14_yellow_watermelon': ('red-fleshed watermelon', False, 'object present with typical colour'),
    'attr_15_wooden_fork': ('fried rice in a pan with a wooden spatula; no fork', False, 'a wooden utensil is present; fork vs spatula is ambiguous'),
    'attr_16_clear_keyboard': ('opaque white keyboard', False, 'object present with typical material'),
    'attr_17_circular_brick': ('living room with sofa, plant, dresser, lamp; no paving blocks (same pixels as abs_14)', True, ['paving blocks', 'paving block', 'bricks', 'brick']),
    'attr_18_red_banana': ('a single pineapple on white; no banana', True, ['banana', 'bananas']),
    'abs_03_empty_plate': ('four people holding drinks by a mural; no plate', True, ['plate', 'dinner plate']),
    'abs_04_empty_cup': ('cup of latte on a saucer', False, 'cup is full; relabel gives an ordinary question, not a trap'),
    'abs_05_empty_leash': ('dog eating from a food bag', False, 'dog present; leash presence not decidable'),
    'abs_07_empty_wrist': ('head-and-shoulders portrait; no wrist or watch visible', True, ['wristwatch', 'watch']),
    'abs_08_empty_bookshelf': ('notepads, pencil, phone calculator and coins on a table; no shelf or book', True, ['book', 'bookshelf', 'shelf']),
    'abs_09_empty_hanger': ('bedroom with bed and bench', False, 'a garment may lie on the bed; hanger absence not certain'),
    'abs_10_empty_birdcage': ('castle on a hill above fog; no cage or bird (same pixels as attr_13)', True, ['bird', 'cage', 'birdcage']),
    'abs_11_empty_vase': ('stacked plates and bowls on a wooden shelf; no vase', True, ['vase', 'flowers']),
    'abs_12_empty_parking_bay': ('aerial view of a partly occupied car park', False, 'target space unspecified'),
    'abs_14_empty_pedestal': ('living room; dried grass in a vase on a dresser; no pedestal (same pixels as attr_17)', True, ['pedestal', 'sculpture']),
    'abs_15_empty_easel': ('paintbrushes and paint on a painted surface', False, 'paint/canvas present; "painting" not clearly absent'),
    'abs_16_empty_mailbox': ('yellow and pink painted wall', True, ['mailbox', 'letters']),
    'abs_17_empty_dogbowl': ('two guinea pigs eating grated carrot from the floor; no bowl', True, ['bowl', 'pet bowl']),
    'abs_18_empty_hook': ('gallery wall of framed pictures on shelves; no hook or keys', True, ['keys', 'hook']),
    'ctrl_02_real_mouse_on_desk': ('computer mouse on a black surface', False, 'mouse present but "desk" premise unverified; needs prompt change, not relabel'),
    'ctrl_03_real_stop_sign': ('laptop showing a 404 page; no sign', True, ['stop sign', 'sign']),
    'ctrl_10_real_clock': ('latte cups and plants; no clock', True, ['clock']),
}


def main():
    audit = {r['id']: r for r in json.loads(Path('dataset/audit_2026-10-30.json').read_text())}
    bench = {r['id']: r for r in json.loads(Path('dataset/counterprior_benchmark.json').read_text())}
    corrected = json.loads(Path('dataset/counterprior_corrected.json').read_text())
    excluded = [i for i, r in audit.items() if r['decision'] == 'exclude']
    assert sorted(excluded) == sorted(JUDGMENTS), set(excluded) ^ set(JUDGMENTS)
    with open('dataset/exclusion_review_2026-10-30.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'category', 'exclusion_reason', 'what_image_shows', 'fixable_by_relabel', 'relabel_or_reason', 'reviewer'])
        for i in sorted(excluded):
            shows, fixable, detail = JUDGMENTS[i]
            note = ('counter_existence; absence_targets=' + '|'.join(detail)) if fixable else detail
            w.writerow([i, bench[i]['category'], audit[i]['pixel_observation'], shows, 'yes' if fixable else 'no', note, REVIEWER])
    proposed = [dict(c, relabel=None) for c in corrected]
    for i in sorted(excluded):
        shows, fixable, targets = JUDGMENTS[i]
        if not fixable:
            continue
        row = dict(bench[i])
        row.update(category='counter_existence', expected_truth=f'There is no {targets[0]} in the image',
                   absence_targets=targets, image_group=audit[i]['pixel_sha256'], image_sha256=audit[i]['file_sha256'],
                   relabel=dict(from_category=bench[i]['category'], from_expected_truth=bench[i]['expected_truth'],
                                reason=shows, reviewer=REVIEWER, status='proposed_needs_human_confirmation'))
        proposed.append(row)
    Path('dataset/counterprior_relabel_proposed_2026-10-31.json').write_text(json.dumps(proposed, indent=2) + '\n')
    summary = dict(excluded=len(excluded), fixable=sum(JUDGMENTS[i][1] for i in excluded),
                   current_rows=len(corrected), proposed_rows=len(proposed),
                   current_category_counts=dict(Counter(c['category'] for c in corrected)),
                   proposed_category_counts=dict(Counter(c['category'] for c in proposed)),
                   proposed_image_groups=len({c['image_group'] for c in proposed}))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
