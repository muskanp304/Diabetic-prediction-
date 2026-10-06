"""
Generates a downloadable Clinical Diagnostic Summary Report PDF for Patients
"""

import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_patient_pdf(patient_name, inputs, is_diabetic, severity, prob, factors, actions):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_color = '#dc2626' if (is_diabetic and severity == 'High') else ('#ea580c' if (is_diabetic and severity == 'Moderate') else ('#d97706' if is_diabetic else '#16a34a'))

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0284c7'), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#64748b'), spaceAfter=12
    )
    section_style = ParagraphStyle(
        'SecHeader', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0f172a'), spaceBefore=8, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        'DocNorm', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#334155'), leading=11
    )
    bold_style = ParagraphStyle(
        'DocBld', parent=styles['Normal'], fontSize=8.5, fontName='Helvetica-Bold', textColor=colors.HexColor('#0f172a')
    )

    story = []
    story.append(Paragraph("PIMA CLINICAL DIABETES EVALUATION & RISK REPORT", title_style))
    story.append(Paragraph("Computer-Aided Diagnostic Assessment & Action Protocol", subtitle_style))

    # Patient info
    p_info = [
        [Paragraph("Patient Name:", bold_style), Paragraph(str(patient_name), normal_style),
         Paragraph("Assessment Date:", bold_style), Paragraph("06-Oct-2026", normal_style)],
        [Paragraph("Age:", bold_style), Paragraph(f"{inputs.get('Age', '-')} Years", normal_style),
         Paragraph("Pregnancies:", bold_style), Paragraph(f"{inputs.get('Pregnancies', '-')}", normal_style)]
    ]
    t_info = Table(p_info, colWidths=[100, 160, 100, 180])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 10))

    # Diagnostic Result Banner
    if is_diabetic:
        diag_title = f"DIABETES PREDICTED: {severity.upper()} SEVERITY"
        diag_bg = '#fee2e2' if severity == 'High' else ('#ffedd5' if severity == 'Moderate' else '#fef3c7')
        diag_border = '#ef4444' if severity == 'High' else ('#f97316' if severity == 'Moderate' else '#f59e0b')
        diag_text = (
            f"<b>Final Clinical Status:</b> POSITIVE FOR DIABETES<br/>"
            f"<b>Assigned Severity Grade:</b> <font color='{title_color}'><b>{severity.upper()} SEVERITY</b></font><br/>"
            f"<b>Model Prediction Probability:</b> {prob*100:.1f}%"
        )
    else:
        diag_title = "DIABETES NOT PREDICTED (NORMAL STATUS)"
        diag_bg = '#ecfdf5'
        diag_border = '#10b981'
        diag_text = (
            f"<b>Final Clinical Status:</b> NEGATIVE / NO DIABETES DETECTED<br/>"
            f"<b>Model Prediction Probability:</b> {prob*100:.1f}% (Low Risk Baseline)<br/>"
            f"<b>Recommendation:</b> Maintain routine active lifestyle and annual checkups."
        )

    t_res = Table([[Paragraph(f"<font color='{title_color}' size='12'><b>{diag_title}</b></font><br/>{diag_text}", normal_style)]], colWidths=[540])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(diag_bg)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(diag_border)),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    # Biomarkers Table
    story.append(Paragraph("EVALUATED CLINICAL BIOMARKERS", section_style))
    b_data = [
        [Paragraph("Biomarker Parameter", bold_style), Paragraph("Evaluated Value", bold_style), Paragraph("Clinical Threshold", bold_style)]
    ]
    param_meta = [
        ("Plasma Fasting Glucose", f"{inputs.get('Glucose', '-')} mg/dL", "Normal: < 100 | Diabetes: >= 126 mg/dL"),
        ("Diastolic Blood Pressure", f"{inputs.get('BloodPressure', '-')} mmHg", "Normal: < 80 | Hypertension: >= 90 mmHg"),
        ("Body Mass Index (BMI)", f"{inputs.get('BMI', '-')} kg/m²", "Normal: 18.5 - 24.9 | Obese: >= 30 kg/m²"),
        ("Fasting Serum Insulin", f"{inputs.get('Insulin', '-')} mu U/ml", "Normal: 2.6 - 24.9 mu U/ml"),
        ("Triceps Skinfold", f"{inputs.get('SkinThickness', '-')} mm", "Normal: 10 - 25 mm"),
        ("Diabetes Pedigree Function", f"{inputs.get('DiabetesPedigreeFunction', '-')}", "Low: < 0.50 | High Genetic: >= 0.50"),
    ]
    for n, v, t in param_meta:
        b_data.append([Paragraph(n, normal_style), Paragraph(f"<b>{v}</b>", normal_style), Paragraph(t, normal_style)])

    t_bio = Table(b_data, colWidths=[180, 120, 240])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 10))

    if is_diabetic:
        # Action Plan
        story.append(Paragraph(f"STRUCTURED MEDICAL ACTION PLAN ({severity.upper()} SEVERITY)", section_style))
        for act in actions:
            story.append(Paragraph(f"• {act}", normal_style))
            story.append(Spacer(1, 2))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<i>Note: This computer-assisted diagnostic evaluation is based on supervised ML modeling. Please consult a qualified endocrinologist or physician for clinical diagnosis and prescription.</i>", normal_style))

    doc.build(story)
    return buf.getvalue()
