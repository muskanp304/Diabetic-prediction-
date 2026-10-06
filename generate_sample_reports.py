import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

os.makedirs('sample_reports', exist_ok=True)

def create_sample_pdf(filename, patient_name, age, pregnancies, glucose, bp, skin, insulin, bmi, pedigree, remarks):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#0284c7'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=14
    )
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=8
    )
    normal_style = ParagraphStyle(
        'DocNormal',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#334155')
    )
    bold_style = ParagraphStyle(
        'DocBold',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0f172a')
    )

    story = []
    
    story.append(Paragraph("APOLLO DIAGNOSTICS & METABOLIC CARE CENTER", title_style))
    story.append(Paragraph("Comprehensive Clinical Pathology & Diabetes Diagnostic Evaluation Report", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Patient Demographics Table
    demo_data = [
        [Paragraph("Patient Name:", bold_style), Paragraph(patient_name, normal_style),
         Paragraph("Date of Test:", bold_style), Paragraph("06-Oct-2026", normal_style)],
        [Paragraph("Patient Age:", bold_style), Paragraph(f"{age} Years", normal_style),
         Paragraph("Gender / Parity:", bold_style), Paragraph(f"Female (Gravida: {pregnancies})", normal_style)],
        [Paragraph("Referred By:", bold_style), Paragraph("Dr. R. Sharma, MD Diabetology", normal_style),
         Paragraph("Sample Type:", bold_style), Paragraph("Fluoride Plasma & Serum", normal_style)]
    ]
    t_demo = Table(demo_data, colWidths=[100, 160, 100, 180])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 14))

    # Diagnostic Test Parameters Table
    story.append(Paragraph("BIOCHEMICAL INVESTIGATION RESULTS", header_style))
    
    table_data = [
        [Paragraph("Test Investigation", bold_style),
         Paragraph("Observed Result", bold_style),
         Paragraph("Standard Biological Unit", bold_style),
         Paragraph("Reference Interval", bold_style),
         Paragraph("Clinical Status", bold_style)]
    ]
    
    tests = [
        ("Plasma Fasting Glucose", f"{glucose}", "mg/dL", "70.0 - 99.0", "ELEVATED" if glucose >= 140 else "NORMAL"),
        ("Diastolic Blood Pressure", f"{bp}", "mmHg", "60.0 - 80.0", "HIGH" if bp >= 85 else "NORMAL"),
        ("Body Mass Index (BMI)", f"{bmi}", "kg/m²", "18.5 - 24.9", "OBESE" if bmi >= 30 else ("OVERWEIGHT" if bmi >= 25 else "NORMAL")),
        ("Fasting Serum Insulin", f"{insulin}", "mu U/ml", "2.6 - 24.9", "ELEVATED" if insulin >= 100 else "NORMAL"),
        ("Triceps Skinfold Thickness", f"{skin}", "mm", "10.0 - 25.0", "ABOVE NORMAL" if skin >= 28 else "NORMAL"),
        ("Diabetes Pedigree Function (DPF)", f"{pedigree}", "Index Score", "0.05 - 0.50", "HIGH RISK" if pedigree >= 0.5 else "LOW RISK"),
        ("Pregnancies (Gravida)", f"{pregnancies}", "Count", "0 - 10", "Documented")
    ]
    
    for t_name, val, unit, ref, status in tests:
        status_color = '#dc2626' if ('ELEVATED' in status or 'HIGH' in status or 'OBESE' in status) else '#16a34a'
        status_p = Paragraph(f"<font color='{status_color}'><b>{status}</b></font>", normal_style)
        table_data.append([
            Paragraph(t_name, normal_style),
            Paragraph(f"<b>{val}</b>", normal_style),
            Paragraph(unit, normal_style),
            Paragraph(ref, normal_style),
            status_p
        ])
        
    t_tests = Table(table_data, colWidths=[160, 90, 110, 100, 80])
    t_tests.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0f2fe')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_tests)
    story.append(Spacer(1, 14))

    story.append(Paragraph("CLINICAL PATHOLOGIST OBSERVATIONS", header_style))
    story.append(Paragraph(remarks, normal_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<i>Electronically authenticated report. Generated for clinical diagnostic risk evaluation.</i>", normal_style))

    doc.build(story)
    print(f"Generated: {filename}")

# 1. High Risk Report
create_sample_pdf(
    'sample_reports/patient_report_high_risk.pdf',
    patient_name="Sunita Sharma",
    age=52,
    pregnancies=6,
    glucose=185,
    bp=88,
    skin=36,
    insulin=195,
    bmi=36.4,
    pedigree=0.85,
    remarks="Patient exhibits significant hyperglycemia (Glucose: 185 mg/dL) combined with Class II obesity (BMI: 36.4) and elevated insulin resistance. High hereditary diabetes pedigree noted. Urgent medical intervention and diabetologist review indicated."
)

# 2. Moderate Risk Report
create_sample_pdf(
    'sample_reports/patient_report_moderate_risk.pdf',
    patient_name="Pooja Verma",
    age=42,
    pregnancies=3,
    glucose=152,
    bp=80,
    skin=30,
    insulin=130,
    bmi=31.2,
    pedigree=0.58,
    remarks="Observed impaired fasting glucose (152 mg/dL) along with moderate obesity index (BMI: 31.2). Mildly elevated insulin resistance. Formal physician consultation and therapeutic lifestyle intervention advised."
)

# 3. Low Severity / Early Stage Diabetic Report
create_sample_pdf(
    'sample_reports/patient_report_low_risk.pdf',
    patient_name="Anita Desai",
    age=31,
    pregnancies=7,
    glucose=103,
    bp=66,
    skin=32,
    insulin=0,
    bmi=39.1,
    pedigree=0.344,
    remarks="Early glycemic irregularity observed with Class II obesity (BMI: 39.1) and multiple parity. Fasting glucose at borderline metabolism. Early metabolic shift detected. Lifestyle modification, daily exercise, and dietary fiber increase strongly advised."
)

# 4. Normal / Non-Diabetic Report
create_sample_pdf(
    'sample_reports/patient_report_normal.pdf',
    patient_name="Priya Patel",
    age=28,
    pregnancies=1,
    glucose=92,
    bp=70,
    skin=20,
    insulin=45,
    bmi=22.5,
    pedigree=0.25,
    remarks="All glycemic and metabolic markers within normal physiological reference ranges. Fasting glucose (92 mg/dL) and BMI (22.5) optimal. Routine annual checkups recommended."
)

# Also create sample CSV
import pandas as pd
csv_data = [
    {"PatientName": "Sunita Sharma", "Age": 52, "Pregnancies": 6, "Glucose": 185, "BloodPressure": 88, "SkinThickness": 36, "Insulin": 195, "BMI": 36.4, "DiabetesPedigreeFunction": 0.85},
    {"PatientName": "Pooja Verma", "Age": 42, "Pregnancies": 3, "Glucose": 152, "BloodPressure": 80, "SkinThickness": 30, "Insulin": 130, "BMI": 31.2, "DiabetesPedigreeFunction": 0.58},
    {"PatientName": "Anita Desai", "Age": 31, "Pregnancies": 7, "Glucose": 103, "BloodPressure": 66, "SkinThickness": 32, "Insulin": 0, "BMI": 39.1, "DiabetesPedigreeFunction": 0.344},
    {"PatientName": "Priya Patel", "Age": 28, "Pregnancies": 1, "Glucose": 92, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 45, "BMI": 22.5, "DiabetesPedigreeFunction": 0.25}
]
pd.DataFrame(csv_data).to_csv('sample_reports/sample_patient_records.csv', index=False)
print("Generated: sample_reports/sample_patient_records.csv")
