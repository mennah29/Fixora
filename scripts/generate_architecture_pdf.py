import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.pdfgen import canvas

# =============================================================================
# 1. Generate High-Resolution Architecture Flowchart Image
# =============================================================================
def generate_diagram_image(output_path="D:/fixora/Fixora_Pipeline_Diagram.png"):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Brand Colors
    c_navy = '#0b496b'
    c_teal = '#188a72'
    c_ice = '#cae7e8'
    c_mint = '#d3f0cb'
    c_red = '#D92D20'
    c_dark = '#071923'
    c_card = '#f4f9fa'

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, text_color='#071923', bold_title=True):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=11, 
                fontweight='bold' if bold_title else 'normal', color=text_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=8.5, 
                    color='#4a6878', zorder=3)

    def draw_arrow(x1, y1, x2, y2, color=c_teal):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16), zorder=1)

    # Title Banner
    ax.text(7, 9.6, "FIXORA — INDUSTRIAL AI DIAGNOSTIC PIPELINE", ha='center', va='center',
            fontsize=16, fontweight='bold', color=c_navy)
    ax.text(7, 9.2, "End-to-End Grounded Retrieval-Augmented Generation & Safety Architecture", 
            ha='center', va='center', fontsize=10, color=c_teal)

    # Level 1: Input
    draw_box(0.8, 7.6, 5.8, 1.1, "1A. Voice Microphone Input", "Browser Speech VAD / Hands-Free Utterance", c_card, c_ice, c_navy)
    draw_box(7.4, 7.6, 5.8, 1.1, "1B. ChatGPT Chat Input & Suggestion Pills", "Text Symptom & Equipment Manual Selector", c_card, c_ice, c_navy)

    # Level 2: STT
    draw_box(0.8, 5.9, 5.8, 1.1, "2. Speech-to-Text (STT) Layer", "Web Speech API (Browser) OR openai/whisper-large-v3-turbo", c_mint, c_teal, c_dark)
    draw_arrow(3.7, 7.6, 3.7, 7.0)

    # Level 3: Hybrid Retrieval
    draw_arrow(3.7, 5.9, 5.5, 5.2)
    draw_arrow(10.3, 7.6, 8.5, 5.2)
    
    draw_box(2.0, 4.0, 10.0, 1.2, "3. Hybrid Manual Retrieval Engine (ChromaDB Vector Store)", 
             "• all-MiniLM-L6-v2 Embeddings (384-dim)  • 8,117 Chunks Indexed  • Exact Code Regex Matching", 
             '#e8f4f7', c_navy, c_navy)

    # Level 4: Groq LLM Reasoning
    draw_arrow(7.0, 4.0, 7.0, 3.3)
    draw_box(2.0, 2.2, 10.0, 1.1, "4. Groq LPU Reasoning Engine (qwen/qwen3.8-27b)", 
             "Senior Maintenance Engineer Persona • Strict Manual Grounding • Structured JSON Checklist", 
             c_mint, c_teal, c_dark)

    # Level 5: Safety Guardrails
    draw_arrow(4.5, 2.2, 3.2, 1.5)
    draw_arrow(9.5, 2.2, 10.8, 1.5)

    draw_box(0.8, 0.4, 5.8, 1.1, "5A. Critical Safety & Hazard Guardrail", 
             "High-Voltage / LOTO Detector -> Red Warning Banner", '#fee4e2', c_red, c_red)
    draw_box(7.4, 0.4, 5.8, 1.1, "5B. Dual Presentation & Speech Synthesis", 
             "Procedure Cards + Page Citations + Browser TTS Speech", c_card, c_teal, c_navy)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Diagram saved to {output_path}")

# =============================================================================
# 2. Numbered Canvas for PDF Header / Footer
# =============================================================================
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Top Accent Header Bar
        self.setFillColor(colors.HexColor('#0b496b'))
        self.rect(0, 772, 612, 20, fill=True, stroke=False)
        self.setFillColor(colors.HexColor('#188a72'))
        self.rect(0, 768, 612, 4, fill=True, stroke=False)
        
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 8)
        self.drawString(36, 778, "FIXORA — INDUSTRIAL AI ASSISTANT · SYSTEM ARCHITECTURE SPECIFICATION")
        
        # Bottom Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#52798e'))
        self.drawString(36, 24, "Confidential · Field Service Engineering Documentation")
        self.drawRightString(576, 24, f"Page {self._pageNumber} of {page_count}")
        self.setStrokeColor(colors.HexColor('#cae7e8'))
        self.setLineWidth(0.8)
        self.line(36, 36, 576, 36)
        self.restoreState()

# =============================================================================
# 3. Generate Complete Architecture PDF Document
# =============================================================================
def generate_architecture_pdf(pdf_path="D:/fixora/Fixora_Architecture_and_Pipeline.pdf"):
    diagram_img_path = "D:/fixora/Fixora_Pipeline_Diagram.png"
    generate_diagram_image(diagram_img_path)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0b496b'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#188a72'),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0b496b'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor('#071923'),
        spaceAfter=6
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0b496b')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Fixora System Architecture & Pipeline Specification", title_style))
    story.append(Paragraph("Comprehensive Technical Reference · AI Models, Hybrid Retrieval & Safety Guardrails", subtitle_style))
    
    # Overview
    story.append(Paragraph(
        "<b>Fixora</b> is an industrial AI maintenance copilot engineered for biomedical and field service engineers. "
        "It converts multi-thousand-page technical service manuals into an instant, conversational, voice-enabled troubleshooting workspace. "
        "The architecture enforces deterministic grounding over <b>8,117 ChromaDB manual chunks</b> and leverages ultra-low latency Groq LPU inference.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Architecture Diagram Image
    if os.path.exists(diagram_img_path):
        story.append(Paragraph("<b>1. High-Level Architectural Flowchart</b>", h2_style))
        img = Image(diagram_img_path, width=540, height=270)
        story.append(img)
        story.append(Spacer(1, 10))

    # Page Break for Model Inventory Table
    story.append(PageBreak())

    # Section 2: Complete Model Inventory Table
    story.append(Paragraph("<b>2. Complete Inventory of AI Models & Engines</b>", h2_style))
    story.append(Paragraph("The table below details all models, runtimes, and roles across the 6 pipeline stages:", body_style))
    story.append(Spacer(1, 4))

    table_data = [
        [
            Paragraph("<b>Component / Layer</b>", callout_style),
            Paragraph("<b>Model Name</b>", callout_style),
            Paragraph("<b>Provider / Runtime</b>", callout_style),
            Paragraph("<b>Specifications & Function</b>", callout_style)
        ],
        [
            Paragraph("<b>Primary Reasoning LLM</b>", body_style),
            Paragraph("<b>qwen/qwen3.8-27b</b>", body_style),
            Paragraph("Groq LPU Cloud", body_style),
            Paragraph("27B parameters. Senior engineer persona; generates structured JSON checklists in &lt;1.5s.", body_style)
        ],
        [
            Paragraph("<b>Fallback LLM 1</b>", body_style),
            Paragraph("<b>qwen/qwen3.6-27b</b>", body_style),
            Paragraph("Groq LPU Cloud", body_style),
            Paragraph("27B parameters. High-precision technical instruction following fallback.", body_style)
        ],
        [
            Paragraph("<b>Biomedical Fallback</b>", body_style),
            Paragraph("<b>allam-2-7b</b>", body_style),
            Paragraph("Groq LPU Cloud", body_style),
            Paragraph("7B parameters. Multilingual medical and engineering diagnostic fallback.", body_style)
        ],
        [
            Paragraph("<b>Offline Edge Model</b>", body_style),
            Paragraph("<b>Qwen2.5-1.5B-Instruct</b>", body_style),
            Paragraph("Local Ollama / CPU", body_style),
            Paragraph("1.5B parameters (4-bit GGUF) for 100% offline field laptop execution.", body_style)
        ],
        [
            Paragraph("<b>Semantic Embedder</b>", body_style),
            Paragraph("<b>all-MiniLM-L6-v2</b>", body_style),
            Paragraph("sentence-transformers", body_style),
            Paragraph("384-dimensional dense semantic vectors optimized for technical text.", body_style)
        ],
        [
            Paragraph("<b>Vector Store Index</b>", body_style),
            Paragraph("<b>ChromaDB (v0.5+)</b>", body_style),
            Paragraph("Local Embedded Store", body_style),
            Paragraph("<b>8,117 dense chunks</b> indexed with HNSW cosine similarity search.", body_style)
        ],
        [
            Paragraph("<b>Speech-to-Text (Cloud)</b>", body_style),
            Paragraph("<b>Web Speech API</b>", body_style),
            Paragraph("Browser Native", body_style),
            Paragraph("Real-time speech capture with 1.3s Voice Activity Detection (VAD).", body_style)
        ],
        [
            Paragraph("<b>Speech-to-Text (Local)</b>", body_style),
            Paragraph("<b>whisper-large-v3-turbo</b>", body_style),
            Paragraph("Hugging Face / PyTorch", body_style),
            Paragraph("High-fidelity offline transcription from raw uploaded audio files.", body_style)
        ],
        [
            Paragraph("<b>Text-to-Speech (Cloud)</b>", body_style),
            Paragraph("<b>SpeechSynthesis</b>", body_style),
            Paragraph("Browser Native", body_style),
            Paragraph("Instant browser speech synthesis of conversational spoken summaries.", body_style)
        ],
        [
            Paragraph("<b>Text-to-Speech (Local)</b>", body_style),
            Paragraph("<b>facebook/mms-tts-eng</b>", body_style),
            Paragraph("Meta MMS / PyTorch", body_style),
            Paragraph("Local high-quality neural voice synthesizer outputting WAV streams.", body_style)
        ],
        [
            Paragraph("<b>Web Fallback Agent</b>", body_style),
            Paragraph("<b>Tavily Search API</b>", body_style),
            Paragraph("Tavily AI", body_style),
            Paragraph("Domain-restricted search over official manufacturer portals.", body_style)
        ]
    ]

    t = Table(table_data, colWidths=[105, 110, 95, 230])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#cae7e8')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0b496b')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cae7e8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fbfc')])
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Section 3: Safety & Grounding Guardrails
    story.append(Paragraph("<b>3. Safety & Grounding Guardrails</b>", h2_style))
    story.append(Paragraph(
        "Fixora incorporates a dual-tier deterministic safety layer: "
        "<br/>• <b>Hazard Interceptor:</b> Scans user queries and retrieved manual text for high-voltage, radiation, chemical, and pressure keywords (e.g. LOTO, lethal shock). When triggered, a high-priority red warning banner is injected at the top of the UI. "
        "<br/>• <b>Anti-Hallucination Barrier:</b> The LLM is strictly constrained to output <code>NOT_FOUND</code> if the retrieved manual chunks do not contain verified technical evidence, preventing fabricated procedures.",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Publication-quality PDF successfully built at: {pdf_path}")

if __name__ == "__main__":
    generate_architecture_pdf()
