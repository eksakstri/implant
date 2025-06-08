from flask import Flask, request, send_file, jsonify
from PIL import Image
from io import BytesIO
from inpaint import inpaint_image
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inpaint', methods=['POST'])
def inpaint_api():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image file and prompt both are required"}), 400

    image_file = request.files['image']
    prompt = request.form['prompt']

    try:
        image = Image.open(image_file)
        result_img = inpaint_image(image, prompt)

        img_io = BytesIO()
        result_img.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='inpainted.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
