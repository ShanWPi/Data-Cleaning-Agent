from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
from etl.pipeline import run_pipeline
from werkzeug.utils import secure_filename

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{filename}")
            file.save(input_path)

            try:
                result = run_pipeline(input_path, output_path)
                history = result.get("history", []) if isinstance(result, dict) else []
                return render_template("result.html", history=history, output_filename=os.path.basename(output_path))
            except Exception as e:
                flash(str(e))
                return redirect(request.url)

    return render_template("index.html")


@app.route("/download/<filename>")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("File not found")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8501)))
