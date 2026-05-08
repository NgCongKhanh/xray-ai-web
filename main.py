# =========================
# IMPORT
# =========================
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn, shutil, base64
from ultralytics import YOLO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = FastAPI()
model = YOLO("best.pt")   # đặt model cùng thư mục


# =========================
# PREDICT FUNCTION
# =========================
def run_prediction(img_path):
    results = model(img_path)[0]
    boxes = results.boxes

    predictions=[]
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])*100
        name = model.names[cls]
        predictions.append({"name":name,"conf":round(conf,2)})

    results.save("result.jpg")
    return "result.jpg", predictions


# =========================
# GENERATE PDF REPORT
# =========================
def generate_pdf(input_img, result_img, predictions):
    doc = SimpleDocTemplate("report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story=[]

    story.append(Paragraph("Chest X-ray AI Report", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph("Input Image", styles['Heading2']))
    story.append(Image(input_img, width=400, height=300))
    story.append(Spacer(1,15))

    story.append(Paragraph("Detection Result", styles['Heading2']))
    story.append(Image(result_img, width=400, height=300))
    story.append(Spacer(1,15))

    table_data=[["Disease","Confidence %"]]
    for p in predictions:
        table_data.append([p["name"], f"{p['conf']} %"])

    table = Table(table_data)
    story.append(table)

    doc.build(story)


# =========================
# API PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with open("input.jpg","wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_img, predictions = run_prediction("input.jpg")

    with open(result_img,"rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    generate_pdf("input.jpg", result_img, predictions)

    return {"image":encoded, "predictions":predictions}


# =========================
# DOWNLOAD PDF
# =========================
@app.get("/download")
def download_pdf():
    return FileResponse("report.pdf",
                        media_type="application/pdf",
                        filename="Xray_Report.pdf")


# =========================
# HOME PAGE (FULL UI)
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Xray AI</title>

<style>
body{margin:0;font-family:Segoe UI;background:#020617;color:white}
.navbar{display:flex;justify-content:space-between;padding:20px 40px;border-bottom:1px solid #1e293b}
.logo{font-size:22px;font-weight:bold;color:#818cf8}
.hero{text-align:center;padding:40px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px;padding:30px}
.card{background:#1e293b;border-radius:15px;padding:20px;min-height:350px}
.drop-zone{border:2px dashed #475569;border-radius:15px;padding:40px;text-align:center;cursor:pointer}
.btn{width:90%;margin:15px;padding:18px;font-size:20px;background:#475569;border:none;border-radius:10px;color:white;cursor:pointer}
</style>

</head>
<body>

<div class="navbar"><div class="logo">🩻 XrayAI</div></div>
<div class="hero"><h1>Chest X-ray AI Detection</h1></div>

<div class="grid">
<div class="card">
    <div class="drop-zone" id="dropZone">
        Drop Image Here or Click Upload
        <input type="file" id="fileInput" hidden>
    </div>
    <div id="preview"></div>
</div>

<div class="card">
    <div id="result"></div>
    <div id="resultText"></div>
</div>
</div>

<center>
<button class="btn" onclick="predict()">🔍 Predict</button>
<button class="btn" onclick="downloadPDF()">📄 Download PDF</button>
</center>

<script>
let selectedFile=null;
const dropZone=document.getElementById("dropZone");
const fileInput=document.getElementById("fileInput");

dropZone.onclick=()=>fileInput.click();

fileInput.onchange=e=>{
 selectedFile=e.target.files[0];
 const reader=new FileReader();
 reader.onload=e=>{preview.innerHTML=`<img src="${e.target.result}" width=300>`}
 reader.readAsDataURL(selectedFile);
};

async function predict(){
 if(!selectedFile){alert("Upload image first");return;}
 const formData=new FormData();
 formData.append("file",selectedFile);

 const res=await fetch("/predict",{method:"POST",body:formData});
 const data=await res.json();

 document.getElementById("result").innerHTML=`<img src="data:image/jpg;base64,${data.image}" width=300>`;
 let text="";
 data.predictions.forEach(p=>{text+=`<p>${p.name} — <b>${p.conf}%</b></p>`});
 document.getElementById("resultText").innerHTML=text;
}

function downloadPDF(){
 window.location="/download";
}
</script>

</body>
</html>
"""


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)