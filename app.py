
import gradio as gr
from ultralytics import YOLO
from PIL import Image

import sqlite3
import os
import uuid

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from components import HEADER_HTML



os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# LOAD MODEL

model = YOLO("best.pt")

# DATABASE


conn = sqlite3.connect(
    "history.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# PREDICT FUNCTION


def predict_xray(image):

    try:

        
        # CHECK IMAGE
        

        if image is None:

            return (
                None,
                """
                <div class="error-card">
                    ❌ No image uploaded
                </div>
                """,
                None
            )

        # SAVE INPUT IMAGE
       

        unique_id = str(uuid.uuid4())

        input_path = f"uploads/{unique_id}.jpg"

        image = image.convert("RGB")

        image.save(
            input_path,
            quality=95
        )

        # YOLO PREDICT
        

        results = model.predict(
            source=input_path,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        
        # CREATE RESULT IMAGE
        

        plotted = result.plot()

        result_image = Image.fromarray(plotted)

        result_path = f"uploads/result_{unique_id}.jpg"

        result_image.save(result_path)


        diagnosis_html = ""

        predictions = []

       
        # NORMAL CASE
      

        if len(result.boxes) == 0:

            predictions.append(("Normal",100))

            diagnosis_html = """
            <div class="success-card">
                ✅ No abnormality detected
            </div>
            """

            cursor.execute(
                """
                INSERT INTO history(
                    disease,
                    confidence
                )
                VALUES (?,?)
                """,
                (
                    "Normal",
                    100
                )
            )

            conn.commit()

        # DETECTION CASE
    

        else:

            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0]) * 100

                disease = model.names[cls]

                predictions.append((disease, conf))

                diagnosis_html += f"""
                <div class="result-card">

                    <h3>
                         {disease}
                    </h3>

                    <div class="progress-container">

                        <div
                            class="progress-bar"
                            style="width:{conf}%"
                        >
                            {conf:.2f}%
                        </div>

                    </div>

                </div>
                """

                cursor.execute(
                    """
                    INSERT INTO history(
                        disease,
                        confidence
                    )
                    VALUES (?,?)
                    """,
                    (
                        disease,
                        conf
                    )
                )

                conn.commit()

        # PDF REPORT

        pdf_path = f"reports/{unique_id}.pdf"

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        story = []

        # TITLE

        story.append(
            Paragraph(
                "Chest X-ray AI Medical Report",
                styles['Title']
            )
        )

        story.append(Spacer(1,20))

        # INPUT IMAGE

        story.append(
            Paragraph(
                "Input X-ray Image",
                styles['Heading2']
            )
        )

        story.append(
            PDFImage(
                input_path,
                width=400,
                height=300
            )
        )

        story.append(Spacer(1,20))

        # RESULT IMAGE

        story.append(
            Paragraph(
                "AI Detection Result",
                styles['Heading2']
            )
        )

        story.append(
            PDFImage(
                result_path,
                width=400,
                height=300
            )
        )

        story.append(Spacer(1,20))

        # TABLE

        table_data = [["Disease","Confidence %"]]

        for disease, conf in predictions:

            table_data.append([
                disease,
                f"{conf:.2f}%"
            ])

        table = Table(
            table_data,
            colWidths=[250,150]
        )

        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('GRID',(0,0),(-1,-1),1,colors.grey),
        ]))

        story.append(table)

        story.append(Spacer(1,20))

        # BUILD PDF

        doc.build(story)

        # RETURN
        

        return (
            result_path,
            diagnosis_html,
            pdf_path
        )

    # ERROR HANDLER

    except Exception as e:

        print("ERROR:", e)

        return (
            None,
            f"""
            <div class="error-card">
                ❌ ERROR:<br><br>
                {str(e)}
            </div>
            """,
            None
        )

# UI

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=open("style.css").read()
) as app:

    # HEADER

    gr.HTML(HEADER_HTML)

    loading_text = gr.Markdown()

    # IMAGE SECTION
    

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="📤 Upload Chest X-ray"
        )

        output_image = gr.Image(
            label="AI Detection Result"
        )

    diagnosis = gr.HTML(
        label="📋 Diagnosis Result"
    )

    # PDF

    pdf_output = gr.File(
        label="📄 Download PDF Report"
    )


    predict_btn = gr.Button(
        "Peredict",
        variant="primary"
    )

    # BUTTON EVENT
    predict_btn.click(
        fn=lambda:
            "AI is analyzing X-ray...",
        outputs=loading_text
    ).then(
        fn=predict_xray,
        inputs=input_image,
        outputs=[
            output_image,
            diagnosis,
            pdf_output
        ]
    ).then(
        fn=lambda:"",
        outputs=loading_text
    )

app.launch(
    show_error=True
)