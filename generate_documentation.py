#!/usr/bin/env python3
"""
Generate PDF documentation from README for HumToHarmony project.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime

def create_documentation_pdf():
    """Create a professional PDF from the README content."""
    
    # Output file
    output_file = Path(__file__).parent / "HumToHarmony_Documentation.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#FF6B6B',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor='#4ECDC4',
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='#4ECDC4',
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        leftIndent=20,
        spaceAfter=12,
    )
    
    # Add content
    elements.append(Paragraph("🎵 HumToHarmony", title_style))
    elements.append(Paragraph(
        "Music 159 - Computer Programming for Music Applications | UC Berkeley",
        subtitle_style
    ))
    elements.append(Spacer(1, 0.3*inch))
    
    # Purpose
    elements.append(Paragraph("Purpose", heading2_style))
    elements.append(Paragraph(
        """<b>HumToHarmony</b> is a Python-based music production tool that transforms hummed 
        melodies into complete musical compositions—no music theory required. It bridges the 
        gap between having a melody "in your head" and turning it into a finished piece of music.""",
        body_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("The application:", body_style))
    for item in [
        "1. Captures your musical idea through humming",
        "2. Detects pitch and converts it to musical notes in ~1-2 seconds",
        "3. Automatically generates harmony (chords and bass lines)",
        "4. Lets you describe sounds in plain English (e.g., 'warm analog synth')",
        "5. Synthesizes audio and exports as WAV or MIDI"
    ]:
        elements.append(Paragraph(item, body_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Key Features
    elements.append(Paragraph("Key Features & Techniques", heading2_style))
    
    features = [
        ("<b>Hum-to-Melody:</b>", 
         "Converts humming to musical notes in ~1-2 seconds using PYIN pitch detection (fast), "
         "voice isolation bandpass filter (80Hz-1000Hz), Krumhansl-Schmuckler key detection, "
         "and note quantization"),
        ("<b>Auto-Harmony:</b>", 
         "Generates chord progressions & bass lines using rule-based harmony generation "
         "and style templates (Pop, Jazz, Lo-Fi, Classical)"),
        ("<b>Natural Language Timbre:</b>", 
         "Describe sounds in plain English using text parsing and descriptor-to-parameter "
         "mapping (50+ descriptors)"),
        ("<b>Synthesizer Engine:</b>", 
         "Creates professional-quality sounds using subtractive synthesis, ADSR envelopes, "
         "and oscillators (sine, saw, square, triangle)"),
        ("<b>Effects & Mixing:</b>", 
         "Polishes the final output with reverb, delay, chorus, compression, and multi-track mixing")
    ]
    
    for title, desc in features:
        elements.append(Paragraph(f"{title} {desc}", body_style))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Techniques
    elements.append(Paragraph("Techniques from Class", heading2_style))
    class_techniques = [
        "Digital signal processing (waveforms, filtering, envelopes)",
        "Audio analysis (pitch detection using PYIN algorithm, spectral analysis)",
        "MIDI processing and export",
        "Audio effects (comb/allpass filters for reverb, delay lines)",
        "Bandpass filtering for voice isolation"
    ]
    for tech in class_techniques:
        elements.append(Paragraph(f"• {tech}", body_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Techniques Beyond Class", heading2_style))
    beyond_techniques = [
        "Probabilistic YIN (PYIN) pitch detection optimized for monophonic voice (10-20x faster than neural networks)",
        "Voice preprocessing (80Hz-1000Hz bandpass filter, noise gate)",
        "Natural language processing for timbre descriptions",
        "Music theory algorithms (key detection via Krumhansl-Schmuckler, chord progression generation)"
    ]
    for tech in beyond_techniques:
        elements.append(Paragraph(f"• {tech}", body_style))
    
    elements.append(PageBreak())
    
    # How to Use
    elements.append(Paragraph("How to Use", heading2_style))
    
    elements.append(Paragraph("<b>Installation</b>", body_style))
    elements.append(Paragraph(
        '<font name="Courier">cd "final proj"<br/>python -m venv venv<br/>'
        'source venv/bin/activate<br/>pip install -r requirements.txt</font>',
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("<b>Running</b>", body_style))
    elements.append(Paragraph(
        '<font name="Courier">streamlit run app/main.py</font>',
        code_style
    ))
    elements.append(Paragraph("Open your browser to http://localhost:8501", body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("<b>Workflow</b>", body_style))
    workflow_steps = [
        "<b>Upload Audio:</b> Record yourself humming a melody (5-30 seconds). Best results: clear humming in a quiet environment",
        "<b>Analyze Melody:</b> Click 'Analyze Melody' - takes ~1-3 seconds. Voice isolation removes background noise, PYIN detects pitch",
        "<b>Review Notes:</b> See your humming converted to musical notes on a piano roll",
        "<b>Generate Harmony:</b> Select a style (Pop, Jazz, Lo-Fi, etc.) for auto-generated chords and bass",
        "<b>Choose Sounds:</b> Type descriptions like 'bright synth lead' or 'warm piano', or use quick presets",
        "<b>Export:</b> Download as WAV or MIDI"
    ]
    for i, step in enumerate(workflow_steps, 1):
        elements.append(Paragraph(f"{i}. {step}", body_style))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Performance
    elements.append(Paragraph("Performance", heading2_style))
    elements.append(Paragraph(
        """<b>Speed:</b> Typical 10-second humming recording processes in 1-3 seconds total. 
        The PYIN algorithm is 10-20x faster than neural network-based methods while maintaining 
        excellent accuracy for monophonic sources like humming.""",
        body_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    # Assignment Goals
    elements.append(Paragraph("Assignment Goal Alignment", heading2_style))
    goals = [
        ("<b>Creating New Sounds:</b>", "Custom synthesizer with natural language timbre control"),
        ("<b>Transforming Sounds:</b>", "Pitch-to-note conversion, effects processing"),
        ("<b>Classifying Sound:</b>", "Automatic key and tempo detection"),
        ("<b>Supporting Creative Process:</b>", "Complete melody-to-music production pipeline")
    ]
    for goal, impl in goals:
        elements.append(Paragraph(f"{goal} {impl}", body_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')} | UC Berkeley Music 159",
        subtitle_style
    ))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF documentation generated: {output_file}")

if __name__ == "__main__":
    create_documentation_pdf()

