from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import cv2
from alpr.detector import PlateDetector
import pytesseract

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

input_size = 608  # Ajusta el tamaño de entrada según tu modelo
weights_path = f'./alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
iou = 0.45
score = 0.25
detector_patente = PlateDetector(weights_path, input_size=input_size, iou=iou, score=score)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if file.mimetype.startswith('image'):
            result = process_image(filepath)
            return jsonify(result), 200
        elif file.mimetype.startswith('video'):
            result_path = process_video(filepath)
            return jsonify({'result': result_path}), 200
        else:
            return jsonify({'error': 'Formato de archivo no soportado'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>', methods=['GET'])
def get_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        return send_file(filepath), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image(image_path):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("Error al leer la imagen")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = detector_patente.preprocess(frame)
        yolo_out = detector_patente.predict(input_img)
        bboxes = detector_patente.procesar_salida_yolo(yolo_out)
        
        print(f"Número de bboxes detectados: {len(bboxes)}")
        print(f"Bboxes: {bboxes}")
        
        frame_w_preds = detector_patente.draw_bboxes(frame, bboxes)
        
        plates = []
        for i, bbox in enumerate(bboxes):
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
            else:
                print(f"Bbox {i} con formato inesperado: {bbox}")
                continue
            
            plate_img = frame[y:y+h, x:x+w]
            
            gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            plate_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            plate_text = ''.join(e for e in plate_text if e.isalnum())
            
            print(f"Placa {i} detectada. Texto: {plate_text}")
            
            plates.append({
                'bbox': bbox,
                'text': plate_text
            })

        result_frame = cv2.cvtColor(frame_w_preds, cv2.COLOR_RGB2BGR)
        
        temp_filename = f"result_{os.path.basename(image_path)}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        cv2.imwrite(temp_filepath, result_frame)
        
        print(f"Imagen procesada guardada como: {temp_filename}")
        print(f"Número total de placas detectadas y procesadas: {len(plates)}")
        
        return {
            'result_image': temp_filename,
            'plates': plates
        }
    
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return {'error': str(e)}
    
def process_video(video_path):
    try:
        vid = cv2.VideoCapture(video_path)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))

        temp_filename = f"result_{os.path.basename(video_path)}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_filepath, fourcc, fps, (width, height))
        
        while True:
            success, frame = vid.read()
            if not success:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = detector_patente.preprocess(frame)
            yolo_out = detector_patente.predict(input_img)
            bboxes = detector_patente.procesar_salida_yolo(yolo_out)
            frame_w_preds = detector_patente.draw_bboxes(frame, bboxes)

            # Imprimir resultados en consola
            print("Resultados del video:")
            for bbox in bboxes:
                print(f"Bbox: {bbox}")
            
            result_frame = cv2.cvtColor(frame_w_preds, cv2.COLOR_RGB2BGR)
            out.write(result_frame)
        
        vid.release()
        out.release()
        
        return temp_filename
    
    except Exception as e:
        print(f"Error procesando el video: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
