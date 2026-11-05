"""Build the public technical protocol PDF from the same versioned site export."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4


def main():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'site/src/data/research.json').read_text(encoding='utf8'))
    target=root/'site/public/technical-report.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ResearchTitle',fontName='Helvetica',fontSize=25,leading=30,spaceAfter=24,textColor=colors.HexColor('#202823')))
    styles.add(ParagraphStyle(name='ResearchBody',fontName='Helvetica',fontSize=9.5,leading=14,spaceAfter=10,textColor=colors.HexColor('#202823')))
    styles.add(ParagraphStyle(name='ResearchHead',fontName='Helvetica-Bold',fontSize=13.5,leading=18,spaceBefore=16,spaceAfter=8,textColor=colors.HexColor('#285e47')))
    styles.add(ParagraphStyle(name='ResearchSmall',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=8,textColor=colors.HexColor('#536157')))
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=54,bottomMargin=54,title='CounterPrior: research protocol and evidence status',author='Basavaraj A Naduvinamani',invariant=1)
    story=[]
    def para(text,style='ResearchBody'):story.append(Paragraph(escape(text),styles[style]))
    def heading(text):para(text,'ResearchHead')
    para('COUNTERPRIOR / TECHNICAL RECORD','ResearchSmall')
    para('Detecting and steering object hallucination from hidden states','ResearchTitle')
    para('Basavaraj A Naduvinamani | AI4Bharat Lab, IIT Madras')
    para('Supervisor: Prof. Mitesh M. Khapra. Mentor: Mohammed Safi Ur Rahman Khan.','ResearchSmall')
    heading('Evidence status')
    para('No pretrained-model result has been verified. The initial Colab attempt stopped during setup, before inference. This document describes the frozen evaluation protocol and current evidence boundaries; it is not a peer-reviewed paper or a report of successful hallucination reduction.' if not data['run'] else 'This document accompanies a verified saved run. Read the generated results/REPORT.md and results/REPORT_metrics.json for the complete measured tables and confidence intervals.')
    heading('Research question')
    para('Can a terminal-prompt decoder state predict an incorrect answer on unseen images, and can norm-preserving activation rotation reduce false-yes answers while retaining recognition of present objects? The motivating false-yes question is narrower than the active detector endpoint: the configured detector predicts any wrong answer, including a wrong no.')
    heading('Frozen image partitions')
    rows=[['Partition','Full profile','Quick profile','Used for']]
    purposes={'calibration':'Layer, probe, direction, gate','validation':'Intervention strength','test':'Reporting only'}
    for part in ['calibration','validation','test']:rows.append([part,str(data['protocol']['counts'][part]),str(data['protocol']['quickCounts'][part]),purposes[part]])
    table=Table(rows,colWidths=[79,75,76,257]);table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1e9')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#cdd6ce')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]));story.append(table)
    para('Source: dataset/pope/splits_seed20261030.json and configs/pope_llava.json. These counts are planned image assignments, not evidence that inference completed.','ResearchSmall')
    heading('Readout and intervention')
    para('Capture raw decoder block outputs at the last prompt position before generation, excluding embeddings. Select the linear-probe, mean-difference and PCA-projection layers using image-grouped calibration folds. Answer-token negative log probability and entropy are later first-token baselines, not early-warning features.')
    para('Orient the direction toward correct calibration answers. Choose cosine gates on calibration and intervention strength on validation. SLERP preserves activation norm, whereas addition can change it. Random-direction and shuffled-label controls reuse the selected SLERP strength. Norm preservation alone establishes no factual benefit.')
    story.append(PageBreak())
    if data['run']:
        para('COUNTERPRIOR / MEASURED MODEL EVIDENCE','ResearchSmall')
        para(f"{data['run']['profile']} profile: {data['run']['testImages']} held-out test images",'ResearchHead')
        para(f"Run {data['run']['id']} | {data['run']['hardware']} | Target: {data['run']['target']}",'ResearchSmall')
        para(' '.join(data.get('summary',[])))
        def metric(m):
            if not m or m.get('estimate') is None:return 'Not available'
            value=f"{m['estimate']:.3f}"
            return value+(f" [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]" if m.get('ci95') else ' [CI unavailable]')
        def result_table(rows,widths):
            cells=[[Paragraph(escape(str(c)),styles['ResearchSmall']) for c in row] for row in rows]
            table=Table(cells,colWidths=widths,repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1e9')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#cdd6ce')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.append(table)
        heading('Unsteered baseline')
        result_table([['POPE split','Accuracy [CI]','Recall [CI]','F1 [CI]']]+[[sp,metric(data['results']['baseline']['test:'+sp]['accuracy']),metric(data['results']['baseline']['test:'+sp]['recall']),metric(data['results']['baseline']['test:'+sp]['f1'])] for sp in ['random','popular','adversarial','pooled']],[82,135,135,135])
        heading('Detection on unseen images')
        result_table([['Detector','AUROC [CI]','AUPRC [CI]']]+[[label,metric(data['results']['detection']['detectors'][key]['auroc']),metric(data['results']['detection']['detectors'][key]['auprc'])] for key,label in [('probe','Linear probe'),('mean_difference','Mean difference'),('pca','PCA projection'),('neg_logprob','Answer-token -log p'),('entropy','Answer-token entropy')]],[157,165,165])
        heading('Steering and matched controls')
        result_table([['Method','False-yes delta [CI]','Recall [CI]']]+[[label,metric(data['results']['steering'][key]['delta_false_yes_rate']),metric(data['results']['steering'][key]['recall'])] for key,label in [('slerp_pca','SLERP / PCA'),('add_mean_difference','Addition'),('random_direction','Random direction'),('shuffled_labels','Shuffled labels')]],[157,165,165])
        para('All intervals are image-clustered 95% bootstrap intervals. Rates use the 0-1 scale. Read the full generated report for precision, yes-ratio, retention, validation diagnostics and paired control comparisons.','ResearchSmall')
        para('Source: results/REPORT_metrics.json and results/published_run.json. No weights or source COCO pixel bytes were independently rehashed from this return bundle.','ResearchSmall')
        story.append(PageBreak())
    para('COUNTERPRIOR / EVALUATION AND REPRODUCTION','ResearchSmall')
    heading('Scoring and uncertainty')
    para('Use the official pinned POPE yes/no parser and keep all questions about an image in the same partition, across question sets. Report false-yes rate on absent objects, recall on present objects, and retention among baseline-correct answers as distinct quantities.')
    para(f"Use {data['protocol']['bootstrapIterations']} image-clustered bootstrap resamples with seed {data['protocol']['seed']}. Paired comparisons resample the same images across methods. Intervals are conditional on the frozen calibration and validation choices; they do not establish generalisation beyond the benchmark.")
    heading('Declared interpretation rules')
    para('Detection improvement requires a wholly positive paired interval for probe AUROC minus answer-token negative-log-probability AUROC. Geometry requires the held-out PCA-projection AUROC interval to lie above chance, without implying a universal truth axis. Steering reduction requires a wholly negative paired false-yes delta; recall loss must be stated alongside any reduction. Undefined or crossing intervals remain inconclusive.')
    heading('Corrections and limitations')
    a=data['protocol']['audit'];para(f"Pixel review retained {a['retained']} of {a['original']} custom benchmark rows. The earlier scorer allowed an unrelated negation to override an affirmative absent-object claim. Earlier headline detection and reduction claims are withdrawn because complete valid held-out evidence was not recovered. The corrected custom set is secondary; proposed relabels are not applied.")
    para('POPE is a binary object benchmark, not a general test of long-form hallucination, video, audio or deployed safety. A quick run is exploratory. No external user study, enterprise-risk reduction or production safety benefit is claimed.')
    heading('Modern-model replication')
    para(f"A separate baseline and detection-only replication is registered for {data['modern']['model']}, revision {data['modern']['revision']}. Status: {data['modern']['status']}. The same questions, scorer and splits are required. The adapter is stub-tested; actual checkpoint execution and T4 memory fit remain unverified.")
    heading('Reproduce')
    para('Open notebooks/CounterPrior_Mechanistic_Steering.ipynb on branch claude/site-2026-11 in Colab, select a GPU and run the setup, timing, experiment and report cells. Run files stay on Drive. Keep the configuration fixed after test evaluation; report every executed profile, including negative results.')
    para('CLI: bash scripts/colab_run.sh setup; then use a fixed COUNTERPRIOR_RUN and COUNTERPRIOR_PROFILE=quick for timing, run and report. The headless plotting backend is forced to Agg so Colab cannot pass an unavailable inline backend into the isolated environment.','ResearchSmall')
    para('Code: https://github.com/basavarajnaduvinamani/DA5410-Winter-Project/tree/claude/site-2026-11','ResearchSmall')
    para('Sources: docs/PROTOCOL.md; configs/pope_llava.json; dataset/audit_2026-10-30.json; site/src/data/research.json; results/REPORT.md.','ResearchSmall')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536157'));canvas.drawString(54,30,'CounterPrior | Evidence before claims');canvas.drawRightString(A4[0]-54,30,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)

if __name__=='__main__':main()
