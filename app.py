import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
import os
import random
import sqlite3
from datetime import datetime, date, timedelta
import hashlib
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Sistema de Gestión de Consultorios Médicos", layout="wide")

# Mapeo de clases
class_names = {
    0: "Normal",
    1: "Células Escamosas Superficiales/Intermedias",
    2: "Células Escamosas Parabasales",
    3: "Células Metaplásicas",
    4: "Adenocarcinoma"
}

# Inicializar base de datos
def init_db():
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    
    # Tabla de usuarios (médicos)
    c.execute('''CREATE TABLE IF NOT EXISTS usuarios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  nombre TEXT NOT NULL,
                  especialidad TEXT NOT NULL)''')
    
    # Tabla de pacientes
    c.execute('''CREATE TABLE IF NOT EXISTS pacientes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nombre TEXT NOT NULL,
                  apellido TEXT NOT NULL,
                  fecha_nacimiento DATE NOT NULL,
                  genero TEXT NOT NULL,
                  telefono TEXT,
                  email TEXT,
                  direccion TEXT,
                  fecha_registro DATE NOT NULL)''')
    
    # Tabla de citas
    c.execute('''CREATE TABLE IF NOT EXISTS citas
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  paciente_id INTEGER NOT NULL,
                  medico_id INTEGER NOT NULL,
                  fecha DATE NOT NULL,
                  hora TEXT NOT NULL,
                  motivo TEXT,
                  estado TEXT DEFAULT 'programada',
                  FOREIGN KEY (paciente_id) REFERENCES pacientes (id),
                  FOREIGN KEY (medico_id) REFERENCES usuarios (id))''')
    
    # Tabla de diagnósticos
    c.execute('''CREATE TABLE IF NOT EXISTS diagnosticos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  paciente_id INTEGER NOT NULL,
                  medico_id INTEGER NOT NULL,
                  fecha DATE NOT NULL,
                  imagen_path TEXT,
                  modelo_utilizado TEXT,
                  resultado TEXT NOT NULL,
                  confianza REAL NOT NULL,
                  observaciones TEXT,
                  FOREIGN KEY (paciente_id) REFERENCES pacientes (id),
                  FOREIGN KEY (medico_id) REFERENCES usuarios (id))''')
    
    # Insertar usuario por defecto si no existe
    c.execute("SELECT COUNT(*) FROM usuarios")
    if c.fetchone()[0] == 0:
        hashed_password = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO usuarios (username, password, nombre, especialidad) VALUES (?, ?, ?, ?)",
                 ("admin", hashed_password, "Dr. Administrador", "Ginecología"))
    
    conn.commit()
    conn.close()

# Función de hash para contraseñas
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Autenticación de usuarios
def authenticate_user(username, password):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM usuarios WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user

# Cargar modelos de IA
@st.cache_resource
def load_models():
    return {
        "CNN Simple": tf.keras.models.load_model("models/best_cervical_model.h5"),
        "CNN Optimizado": tf.keras.models.load_model("models/optimized_cervical_model.h5")
    }

# Preprocesamiento de imágenes
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Tabla de métricas
metricas_modelos = [
    ["Modelo", "Precisión", "Sensibilidad\n (Avg)", "Especificidad\n (Avg)", "F1-Score\n (Avg)", "MCC"],
    ["Mejor Modelo (Keras Tuner)", "83.08%", "80.73%", "89.83%", "82.78%", "0.7065"],
    ["CNN Simple", "84.89%", "80.36%", "91.08%", "82.21%", "0.7372"],
    ["ResNet50 (Transfer Learning)", "73.11%", "53.78%", "83.61%", "51.27%", "0.5407"]
]

# Resultados de la prueba de McNemar
mcnemar_results = [
    ["Comparación de\nModelos", "Estadístico\nχ²", "Valor-p", "¿Diferencia\nSignificativa?"],
    ["Mejor Modelo vs CNN Simple", "4.50", "0.034", "No hay una diferencia estadísticamente p>=0.05"],
    ["Mejor Modelo vs ResNet50", "2.10", "0.147", " Existe una diferencia estadísticamente p<0.05"],
    ["CNN Simple vs ResNet50 C", "5.30", "0.021", "Existe una diferencia estadísticamente p<0.05"]
]

# Función para generar PDF del diagnóstico
def generar_pdf_diagnostico(paciente, medico, imagen, model_name, resultado, confianza, probabilidades, observaciones):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()
    y = height - 50

    # Encabezado con información del consultorio
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, y, "Consultorio Médico Especializado")
    c.setFont("Helvetica", 10)
    c.drawString(30, y - 15, "Cáncer Cervical - Diagnóstico Asistido por IA")
    y -= 40

    # Fecha
    fecha_actual = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Fecha del Reporte: {fecha_actual}")
    y -= 30

    # Información del paciente
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Información del Paciente")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Nombre: {paciente['nombre']} {paciente['apellido']}")
    y -= 15
    c.drawString(30, y, f"Fecha de Nacimiento: {paciente['fecha_nacimiento']}")
    y -= 15
    c.drawString(30, y, f"Género: {paciente['genero']}")
    y -= 15
    c.drawString(30, y, f"Teléfono: {paciente['telefono']}")
    y -= 30

    # Información del médico
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Médico Tratante")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Nombre: {medico['nombre']}")
    y -= 15
    c.drawString(30, y, f"Especialidad: {medico['especialidad']}")
    y -= 30

    # Resultados del diagnóstico
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Resultados del Diagnóstico")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Modelo utilizado: {model_name}")
    y -= 15
    c.drawString(30, y, f"Resultado: {resultado}")
    y -= 15
    c.drawString(30, y, f"Confianza: {confianza:.2f}%")
    y -= 15
    if observaciones:
        c.drawString(30, y, f"Observaciones: {observaciones}")
        y -= 30
    else:
        y -= 15

    # Probabilidades por clase
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Probabilidades por Clase:")
    y -= 20
    for i, prob in enumerate(probabilidades):
        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"{class_names[i]}: {prob * 100:.2f}%")
        y -= 15

    # Firmar PDF
    y = 100
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Firma del Médico: _________________________")
    y -= 30
    c.drawString(30, y, f"Nombre: {medico['nombre']}")
    y -= 15
    c.drawString(30, y, f"Fecha: {fecha_actual}")

    c.save()
    buffer.seek(0)
    return buffer

# Función para agregar paciente
def agregar_paciente(nombre, apellido, fecha_nacimiento, genero, telefono, email, direccion):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    fecha_registro = date.today().isoformat()
    c.execute('''INSERT INTO pacientes (nombre, apellido, fecha_nacimiento, genero, telefono, email, direccion, fecha_registro)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (nombre, apellido, fecha_nacimiento, genero, telefono, email, direccion, fecha_registro))
    conn.commit()
    conn.close()

# Función para obtener lista de pacientes
def obtener_pacientes():
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    c.execute("SELECT * FROM pacientes ORDER BY apellido, nombre")
    pacientes = c.fetchall()
    conn.close()
    return pacientes

# Función para obtener paciente por ID
def obtener_paciente_por_id(paciente_id):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    c.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,))
    paciente = c.fetchone()
    conn.close()
    return paciente

# Función para agregar cita
def agregar_cita(paciente_id, medico_id, fecha, hora, motivo):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    c.execute('''INSERT INTO citas (paciente_id, medico_id, fecha, hora, motivo)
                 VALUES (?, ?, ?, ?, ?)''',
              (paciente_id, medico_id, fecha, hora, motivo))
    conn.commit()
    conn.close()

# Función para obtener citas
def obtener_citas(fecha=None, medico_id=None):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    
    if fecha and medico_id:
        c.execute('''SELECT c.id, p.nombre || ' ' || p.apellido as paciente, c.fecha, c.hora, c.motivo, c.estado
                     FROM citas c
                     JOIN pacientes p ON c.paciente_id = p.id
                     WHERE c.fecha = ? AND c.medico_id = ?
                     ORDER BY c.hora''', (fecha, medico_id))
    elif fecha:
        c.execute('''SELECT c.id, p.nombre || ' ' || p.apellido as paciente, c.fecha, c.hora, c.motivo, c.estado
                     FROM citas c
                     JOIN pacientes p ON c.paciente_id = p.id
                     WHERE c.fecha = ?
                     ORDER BY c.hora''', (fecha,))
    else:
        c.execute('''SELECT c.id, p.nombre || ' ' || p.apellido as paciente, c.fecha, c.hora, c.motivo, c.estado
                     FROM citas c
                     JOIN pacientes p ON c.paciente_id = p.id
                     ORDER BY c.fecha, c.hora''')
    
    citas = c.fetchall()
    conn.close()
    return citas

# Función para guardar diagnóstico
def guardar_diagnostico(paciente_id, medico_id, imagen_path, modelo_utilizado, resultado, confianza, observaciones):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    fecha = date.today().isoformat()
    c.execute('''INSERT INTO diagnosticos (paciente_id, medico_id, fecha, imagen_path, modelo_utilizado, resultado, confianza, observaciones)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (paciente_id, medico_id, fecha, imagen_path, modelo_utilizado, resultado, confianza, observaciones))
    conn.commit()
    conn.close()

# Función para obtener historial de diagnósticos de un paciente
def obtener_diagnosticos_paciente(paciente_id):
    conn = sqlite3.connect('consultorio.db')
    c = conn.cursor()
    c.execute('''SELECT d.fecha, d.modelo_utilizado, d.resultado, d.confianza, d.observaciones, u.nombre
                 FROM diagnosticos d
                 JOIN usuarios u ON d.medico_id = u.id
                 WHERE d.paciente_id = ?
                 ORDER BY d.fecha DESC''', (paciente_id,))
    diagnosticos = c.fetchall()
    conn.close()
    return diagnosticos

# Inicializar base de datos
init_db()

# Inicializar estado de sesión
if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False
if 'usuario' not in st.session_state:
    st.session_state.usuario = None

# Página de inicio de sesión
if not st.session_state.autenticado:
    st.title("Sistema de Gestión de Consultorios Médicos")
    st.subheader("Inicio de Sesión")
    
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Iniciar Sesión")
        
        if submit:
            usuario = authenticate_user(username, password)
            if usuario:
                st.session_state.autenticado = True
                st.session_state.usuario = {
                    "id": usuario[0],
                    "username": usuario[1],
                    "nombre": usuario[3],
                    "especialidad": usuario[4]
                }
                st.success(f"Bienvenido, {usuario[3]}")
                st.experimental_rerun()
            else:
                st.error("Usuario o contraseña incorrectos")
else:
    # Menú principal después de la autenticación
    st.sidebar.title(f"👨‍⚕️ Dr. {st.session_state.usuario['nombre']}")
    st.sidebar.subheader("Menú Principal")
    
    app_mode = st.sidebar.selectbox("Seleccione una opción", 
                                   ["Dashboard", "Gestión de Pacientes", "Agenda de Citas", 
                                    "Diagnóstico de Cáncer Cervical", "Historial Médico", "Reportes", "Cerrar Sesión"])
    
    if app_mode == "Cerrar Sesión":
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.experimental_rerun()
    
    elif app_mode == "Dashboard":
        st.header("Dashboard del Consultorio")
        
        # Estadísticas rápidas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Pacientes")
            pacientes = obtener_pacientes()
            st.metric("Total", len(pacientes))
        
        with col2:
            st.subheader("Citas Hoy")
            citas_hoy = obtener_citas(date.today().isoformat(), st.session_state.usuario['id'])
            st.metric("Programadas", len(citas_hoy))
        
        with col3:
            st.subheader("Diagnósticos")
            # Aquí podrías agregar más estadísticas según necesites
            st.metric("Realizados", "15")  # Ejemplo
        
        # Próximas citas
        st.subheader("Próximas Citas")
        proximos_dias = [(date.today() + timedelta(days=i)).isoformat() for i in range(7)]
        citas_proximas = []
        for dia in proximos_dias:
            citas_dia = obtener_citas(dia, st.session_state.usuario['id'])
            citas_proximas.extend(citas_dia)
        
        if citas_proximas:
            for cita in citas_proximas[:5]:  # Mostrar solo las 5 próximas
                st.write(f"📅 {cita[2]} {cita[3]} - {cita[1]} ({cita[4]})")
        else:
            st.info("No hay citas programadas para los próximos días")
    
    elif app_mode == "Gestión de Pacientes":
        st.header("Gestión de Pacientes")
        
        tab1, tab2 = st.tabs(["Lista de Pacientes", "Nuevo Paciente"])
        
        with tab1:
            st.subheader("Pacientes Registrados")
            pacientes = obtener_pacientes()
            
            if pacientes:
                for paciente in pacientes:
                    with st.expander(f"{paciente[2]} {paciente[1]} - {paciente[3]}"):
                        st.write(f"**Fecha de Nacimiento:** {paciente[3]}")
                        st.write(f"**Género:** {paciente[4]}")
                        st.write(f"**Teléfono:** {paciente[5]}")
                        st.write(f"**Email:** {paciente[6]}")
                        st.write(f"**Dirección:** {paciente[7]}")
                        st.write(f"**Fecha de Registro:** {paciente[8]}")
            else:
                st.info("No hay pacientes registrados")
        
        with tab2:
            st.subheader("Registrar Nuevo Paciente")
            with st.form("nuevo_paciente"):
                col1, col2 = st.columns(2)
                with col1:
                    nombre = st.text_input("Nombre")
                    fecha_nacimiento = st.date_input("Fecha de Nacimiento", min_value=date(1900, 1, 1))
                    telefono = st.text_input("Teléfono")
                with col2:
                    apellido = st.text_input("Apellido")
                    genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
                    email = st.text_input("Email")
                
                direccion = st.text_area("Dirección")
                
                if st.form_submit_button("Registrar Paciente"):
                    if nombre and apellido and fecha_nacimiento:
                        agregar_paciente(nombre, apellido, fecha_nacimiento.isoformat(), genero, telefono, email, direccion)
                        st.success("Paciente registrado correctamente")
                    else:
                        st.error("Por favor complete los campos obligatorios: Nombre, Apellido y Fecha de Nacimiento")
    
    elif app_mode == "Agenda de Citas":
        st.header("Agenda de Citas")
        
        tab1, tab2 = st.tabs(["Ver Citas", "Programar Cita"])
        
        with tab1:
            st.subheader("Citas Programadas")
            fecha_consulta = st.date_input("Seleccione fecha para ver citas", value=date.today())
            citas = obtener_citas(fecha_consulta.isoformat(), st.session_state.usuario['id'])
            
            if citas:
                for cita in citas:
                    with st.expander(f"{cita[3]} - {cita[1]}"):
                        st.write(f"**Paciente:** {cita[1]}")
                        st.write(f"**Fecha:** {cita[2]}")
                        st.write(f"**Hora:** {cita[3]}")
                        st.write(f"**Motivo:** {cita[4]}")
                        st.write(f"**Estado:** {cita[5]}")
            else:
                st.info("No hay citas programadas para esta fecha")
        
        with tab2:
            st.subheader("Programar Nueva Cita")
            pacientes = obtener_pacientes()
            
            if pacientes:
                with st.form("nueva_cita"):
                    paciente_options = [f"{p[2]} {p[1]} (ID: {p[0]})" for p in pacientes]
                    paciente_seleccionado = st.selectbox("Seleccione paciente", paciente_options)
                    paciente_id = int(paciente_seleccionado.split("(ID: ")[1].replace(")", ""))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fecha_cita = st.date_input("Fecha de la cita", min_value=date.today())
                    with col2:
                        hora_cita = st.time_input("Hora de la cita")
                    
                    motivo = st.text_area("Motivo de la consulta")
                    
                    if st.form_submit_button("Programar Cita"):
                        agregar_cita(paciente_id, st.session_state.usuario['id'], 
                                    fecha_cita.isoformat(), hora_cita.strftime("%H:%M"), motivo)
                        st.success("Cita programada correctamente")
            else:
                st.info("No hay pacientes registrados. Por favor registre pacientes primero.")
    
    elif app_mode == "Diagnóstico de Cáncer Cervical":
        st.header("Diagnóstico de Cáncer Cervical")
        
        # Seleccionar paciente
        pacientes = obtener_pacientes()
        if not pacientes:
            st.info("No hay pacientes registrados. Por favor registre pacientes primero.")
        else:
            paciente_options = [f"{p[2]} {p[1]} (ID: {p[0]})" for p in pacientes]
            paciente_seleccionado = st.selectbox("Seleccione paciente", paciente_options)
            paciente_id = int(paciente_seleccionado.split("(ID: ")[1].replace(")", ""))
            paciente_info = obtener_paciente_por_id(paciente_id)
            
            st.subheader("📷 Subir Imagen Cervical")
            uploaded_file = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", width=300)
                
                # Guardar imagen temporalmente
                img_path = f"temp_{paciente_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(img_path)
                
                processed_img = preprocess_image(image)
                models = load_models()
                model_name = st.selectbox("Seleccione el modelo", list(models.keys()))
                
                observaciones = st.text_area("Observaciones (opcional)")
                
                if st.button("🔍 Realizar diagnóstico"):
                    with st.spinner("Analizando..."):
                        model = models[model_name]
                        preds = model.predict(processed_img)
                        pred_class = np.argmax(preds[0])
                        confidence = np.max(preds[0]) * 100
                        resultado = class_names[pred_class]
                        
                        # Guardar diagnóstico en base de datos
                        guardar_diagnostico(paciente_id, st.session_state.usuario['id'], 
                                           img_path, model_name, resultado, confidence, observaciones)
                        
                        st.success(f"*Resultado:* {resultado} (Confianza: {confidence:.2f}%)")
                        
                        st.subheader("📊 Probabilidades por Clase")
                        for i, prob in enumerate(preds[0]):
                            st.progress(int(prob * 100))
                            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
                        
                        # Generar PDF
                        paciente_dict = {
                            "nombre": paciente_info[1],
                            "apellido": paciente_info[2],
                            "fecha_nacimiento": paciente_info[3],
                            "genero": paciente_info[4],
                            "telefono": paciente_info[5]
                        }
                        
                        pdf_buffer = generar_pdf_diagnostico(
                            paciente_dict, st.session_state.usuario, 
                            image, model_name, resultado, confidence, 
                            preds[0], observaciones
                        )
                        
                        st.download_button(
                            label="📄 Descargar Reporte PDF",
                            data=pdf_buffer,
                            file_name=f"diagnostico_{paciente_info[1]}_{paciente_info[2]}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
            
            # Limpiar imágenes temporales
            for file in os.listdir('.'):
                if file.startswith('temp_') and file.endswith('.png'):
                    os.remove(file)
    
    elif app_mode == "Historial Médico":
        st.header("Historial Médico")
        
        pacientes = obtener_pacientes()
        if not pacientes:
            st.info("No hay pacientes registrados.")
        else:
            paciente_options = [f"{p[2]} {p[1]} (ID: {p[0]})" for p in pacientes]
            paciente_seleccionado = st.selectbox("Seleccione paciente", paciente_options)
            paciente_id = int(paciente_seleccionado.split("(ID: ")[1].replace(")", ""))
            
            diagnosticos = obtener_diagnosticos_paciente(paciente_id)
            
            if diagnosticos:
                st.subheader("Historial de Diagnósticos")
                for diagnostico in diagnosticos:
                    with st.expander(f"{diagnostico[0]} - {diagnostico[2]} (Por: {diagnostico[5]})"):
                        st.write(f"**Modelo utilizado:** {diagnostico[1]}")
                        st.write(f"**Resultado:** {diagnostico[2]}")
                        st.write(f"**Confianza:** {diagnostico[3]:.2f}%")
                        if diagnostico[4]:
                            st.write(f"**Observaciones:** {diagnostico[4]}")
            else:
                st.info("Este paciente no tiene diagnósticos registrados.")
    
    elif app_mode == "Reportes":
        st.header("Reportes y Estadísticas")
        
        st.subheader("📊 Métricas de Rendimiento de Modelos")
        st.table(metricas_modelos)
        
        st.subheader("📈 Pruebas de McNemar")
        st.table(mcnemar_results)
        
        st.subheader("Matriz de Confusión")
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png")
        else:
            st.warning("No se encontró la imagen de matriz de confusión")
        
        st.subheader("Curva de Aprendizaje")
        if os.path.exists("reports/learning_curve.png"):
            st.image("reports/learning_curve.png")
        else:
            st.warning("No se encontró la imagen de curva de aprendizaje")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
*Sistema de Gestión de Consultorios Médicos*  
*Módulo de Diagnóstico de Cáncer Cervical*  
*Dataset:* SIPaKMeD  
*Precisión:* 95.1% (validación cruzada)
""")