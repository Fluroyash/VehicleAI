import os
import tempfile
import subprocess

import cv2
import pandas as pd
import streamlit as st

from ultralytics import YOLO


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VehicleAI - Traffic Analytics",
    page_icon="🚗",
    layout="wide"
)



# Modern dark dashboard styling. Keeps the existing detection logic intact.
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --vai-bg: #080d19;
        --vai-panel: #111b2d;
        --vai-panel-2: #17243a;
        --vai-border: rgba(148, 173, 205, 0.22);
        --vai-text: #edf5ff;
        --vai-muted: #9db0ca;
        --vai-cyan: #42d9f5;
        --vai-green: #5ee6a8;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #080d19 !important;
        color: var(--vai-text) !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(31, 95, 130, .24), transparent 38%),
            radial-gradient(ellipse at 92% 8%, rgba(42, 92, 163, .20), transparent 32%),
            #080d19 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 25, .92) !important;
    }

    [data-testid="stSidebar"] {
        background: #0d1627 !important;
        border-right: 1px solid var(--vai-border) !important;
    }
    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--vai-text);
    }

    .block-container {
        max-width: 1500px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
    }

    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid rgba(66, 217, 245, .32);
        border-radius: 24px;
        background: linear-gradient(120deg, #132943 0%, #111b2e 65%, #10263a 100%);
        box-shadow: 0 18px 60px rgba(0, 0, 0, .28);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -110px;
        border-radius: 50%;
        border: 1px solid rgba(66, 217, 245, .28);
        box-shadow: 0 0 0 28px rgba(66, 217, 245, .05),
                    0 0 0 58px rgba(66, 217, 245, .035);
        pointer-events: none;
    }
    .hero-kicker, .section-eyebrow {
        color: var(--vai-cyan) !important;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
    }
    .hero-title {
        font-size: clamp(2.1rem, 4vw, 3rem);
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -.055em;
        margin-top: .4rem;
        color: #f4f8ff !important;
    }
    .hero-title span { color: var(--vai-cyan) !important; }
    .hero-subtitle {
        color: #bdcce0 !important;
        font-size: 1.1rem;
        margin-top: .7rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.3rem;
    }
    .hero-badges span {
        color: #dceafa !important;
        background: rgba(255, 255, 255, .06);
        border: 1px solid rgba(255, 255, 255, .13);
        padding: .4rem .7rem;
        border-radius: 999px;
        font-size: .78rem;
    }

    .section-heading {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 1rem;
        margin: .25rem 0 .8rem;
    }
    .section-heading h2 {
        color: #f4f8ff !important;
        font-size: 1.65rem;
        margin: .2rem 0 0;
        letter-spacing: -.025em;
    }
    .section-description {
        color: var(--vai-muted) !important;
        font-size: .9rem;
        max-width: 370px;
        text-align: right;
    }

    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--vai-text);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #17243a, #111b2d) !important;
        border: 1px solid var(--vai-border) !important;
        border-radius: 16px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .16);
    }
    [data-testid="stMetricLabel"] { color: #a9bdd7 !important; }
    [data-testid="stMetricValue"] { color: #f4f8ff !important; font-weight: 750; }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(66, 217, 245, .52) !important;
        border-radius: 18px !important;
        padding: .7rem !important;
        background: rgba(18, 32, 52, .65) !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(66, 217, 245, .52) !important;
        background: linear-gradient(100deg, #087e9c, #176c88) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        min-height: 2.8rem;
        transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.12);
        border-color: var(--vai-cyan) !important;
        transform: translateY(-1px);
    }

    [data-testid="stAlert"] { border-radius: 12px !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--vai-border);
        border-radius: 12px;
        overflow: hidden;
    }
    hr { border-color: var(--vai-border) !important; }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #131e31 !important;
        border-color: var(--vai-border) !important;
        border-radius: 10px !important;
    }

    .small-note { color: var(--vai-muted) !important; font-size: .82rem; }

    .vai-nav { display:flex; flex-wrap:wrap; gap:.65rem; margin:.25rem 0 1.6rem;
        padding:.8rem; border:1px solid var(--vai-border); border-radius:14px;
        background:rgba(17,27,45,.72); }
    .vai-nav a { display:inline-block; color:#dff7ff !important;
        text-decoration:none !important; padding:.55rem .85rem;
        border:1px solid rgba(66,217,245,.22); border-radius:9px;
        background:rgba(66,217,245,.06); font-weight:650; font-size:.88rem; }
    .vai-nav a:hover { background:rgba(66,217,245,.16); border-color:var(--vai-cyan); }
    .vai-overview-label { color:var(--vai-cyan); text-transform:uppercase;
        letter-spacing:.13em; font-size:.72rem; font-weight:800; margin-bottom:.35rem; }
    .vai-processing-note { padding:.9rem 1rem; margin:.5rem 0 1rem;
        border:1px solid rgba(66,217,245,.25); border-radius:12px;
        background:rgba(17,38,58,.7); color:#d9e9fb; }
    [data-testid="stProgress"] > div > div {
        background:linear-gradient(90deg,#16a6c4,#42d9f5,#5ee6a8);
    }


    /* Professional dashboard layout */
    .vai-topline {
        display:flex; justify-content:space-between; align-items:center;
        gap:1rem; margin:.2rem 0 .8rem; color:var(--vai-muted); font-size:.82rem;
    }
    .vai-status {
        display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .65rem;
        border:1px solid rgba(94,230,168,.25); border-radius:999px;
        color:#8ff0c2; background:rgba(94,230,168,.07); font-weight:700;
    }
    [data-testid="stMetric"] {
        min-height:108px; display:flex; flex-direction:column; justify-content:center;
    }
    [data-testid="stMetricLabel"] p { font-size:.78rem !important; letter-spacing:.02em; }

    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--vai-cyan) !important;
        border-color: var(--vai-cyan) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrack"] {
        background: var(--vai-cyan) !important;
    }

    .vai-section-label {
        color:var(--vai-cyan); text-transform:uppercase; letter-spacing:.13em;
        font-size:.7rem; font-weight:800; margin:.25rem 0 .4rem;
    }

    @media (max-width: 760px) {
        .hero { padding: 1.4rem; }
        .section-heading { display: block; }
        .section-description { text-align: left; margin-top: .5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">COMPUTER VISION · TRAFFIC INTELLIGENCE</div>
        <div class="hero-title">Vehicle<span>AI</span></div>
        <div class="hero-subtitle">
            Real-time vehicle detection, tracking, and traffic analytics
        </div>
        <div class="hero-badges">
            <span>● YOLO26n</span>
            <span>● ByteTrack</span>
            <span>● Traffic density analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="vai-topline"><span>TRAFFIC INTELLIGENCE WORKSPACE</span>'
    '<span class="vai-status">● READY FOR VIDEO ANALYSIS</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-heading">
        <div><span class="section-eyebrow">WORKSPACE</span>
        <h2>Traffic analysis studio</h2></div>
        <div class="section-description">
            Upload a clip, configure detection, and explore the results.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '''
    <nav class="vai-nav">
        <a href="#overview">📊 Overview</a>
        <a href="#input-video">🎥 Detection</a>
        <a href="#analytics">📈 Analytics</a>
        <a href="#results">📋 Results</a>
        <a href="#annotated-video">▶ Annotated video</a>
    </nav>
    ''',
    unsafe_allow_html=True
)


# =========================================================
# 2. SESSION STATE
# =========================================================

if "annotated_video_bytes" not in st.session_state:
    st.session_state["annotated_video_bytes"] = None

if "traffic_results" not in st.session_state:
    st.session_state["traffic_results"] = None


# =========================================================
# 3. SIDEBAR SETTINGS
# =========================================================

st.sidebar.markdown(
    """
    <div style="padding:.6rem 0 1.1rem;">
        <div style="font-size:.7rem;letter-spacing:.16em;color:#42d9f5;font-weight:800;">
            VEHICLEAI
        </div>
        <div style="font-size:1.35rem;font-weight:800;color:#f4f8ff;">
            Control panel
        </div>
        <div style="font-size:.82rem;color:#9aabc4;">
            Detection configuration
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.15,
    step=0.05
)

line_y = st.sidebar.slider(
    "Counting Line Position",
    min_value=50,
    max_value=500,
    value=450,
    step=10
)

st.sidebar.subheader("🚦 Traffic Density Thresholds")

medium_threshold = st.sidebar.number_input(
    "Medium traffic starts at",
    min_value=1,
    max_value=100,
    value=5,
    step=1
)

high_threshold = st.sidebar.number_input(
    "High traffic starts at",
    min_value=2,
    max_value=200,
    value=16,
    step=1
)

if high_threshold <= medium_threshold:
    st.sidebar.error(
        "The high traffic threshold must be greater "
        "than the medium traffic threshold."
    )

st.sidebar.info(
    "Density thresholds are based on the number of visible "
    "vehicles per frame. They are configurable prototype "
    "thresholds, not universal traffic standards."
)

st.sidebar.write("**Model:** YOLO26n")
st.sidebar.write("**Tracking:** ByteTrack")
st.sidebar.write("**Processing:** CPU")
st.sidebar.write("**Resolution:** 960 × 540")


# =========================================================
# 4. LOAD YOLO MODEL
# =========================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"

@st.cache_resource
def load_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        with st.spinner("Downloading YOLO model..."):
            model = YOLO("yolo26n.pt")
    else:
        model = YOLO(str(MODEL_PATH))

    return model

# =========================================================
# 5. VEHICLE CLASSES
# COCO CLASS IDS
# =========================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# =========================================================
# 6. TRAFFIC DENSITY CLASSIFICATION
# =========================================================

def classify_traffic_density(
    average_vehicles,
    medium_limit,
    high_limit
):
    """
    Classify traffic based on the average number of vehicles
    detected per frame.

    These thresholds are configurable prototype settings.
    """

    if average_vehicles < medium_limit:
        return "Low"

    elif average_vehicles < high_limit:
        return "Medium"

    return "High"


def get_density_description(density):
    descriptions = {
        "Low": "Few vehicles are visible in the video.",
        "Medium": "A moderate number of vehicles are visible.",
        "High": "Many vehicles are visible simultaneously."
    }

    return descriptions.get(density, "")


# =========================================================
# 7. VIDEO UPLOADER
# =========================================================

st.markdown('<div id="input-video"></div>', unsafe_allow_html=True)
st.markdown("### 🎥 01 / Input video")
st.caption("Choose a traffic clip to run vehicle detection and analytics.")

uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)


# =========================================================
# 8. H.264 CONVERSION
# =========================================================

def convert_to_browser_video(input_path):
    """
    Convert the annotated video to H.264 MP4 for improved
    browser compatibility.

    If conversion fails, return the original video path.
    """

    output_path = None

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:
            output_path = temp_file.name

        command = [
            ffmpeg_path,
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180
        )

        if (
            result.returncode == 0
            and os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        ):
            return output_path

    except Exception:
        pass

    if output_path and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    return input_path


# =========================================================
# 9. ANALYZE VIDEO
# =========================================================

if uploaded_file is not None:

    st.success(f"Uploaded video: {uploaded_file.name}")

    if high_threshold <= medium_threshold:
        st.warning(
            "Please set the high traffic threshold above "
            "the medium traffic threshold."
        )

    elif st.button(
        "🚀 Analyze Traffic Video",
        type="primary"
    ):

        # Clear results from previous analysis.
        st.session_state["annotated_video_bytes"] = None
        st.session_state["traffic_results"] = None

        input_path = None
        output_path = None
        browser_video_path = None

        cap = None
        writer = None

        try:

            # -------------------------------------------------
            # A. SAVE UPLOADED VIDEO
            # -------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as input_temp:

                input_temp.write(uploaded_file.getvalue())
                input_path = input_temp.name

            # -------------------------------------------------
            # B. LOAD MODEL
            # -------------------------------------------------

            with st.spinner("Loading YOLO model..."):
                model = load_model()

            # -------------------------------------------------
            # C. OPEN INPUT VIDEO
            # -------------------------------------------------

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise RuntimeError(
                    "Could not open the uploaded video. "
                    "Try another video file."
                )

            fps = cap.get(cv2.CAP_PROP_FPS)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if fps <= 0:
                fps = 30.0

            if total_frames <= 0:
                raise RuntimeError(
                    "The uploaded video contains no readable frames."
                )

            original_duration = total_frames / fps

            st.write(
                f"**Video duration:** {original_duration:.2f} seconds"
            )

            st.write(
                f"**Total frames:** {total_frames}"
            )

            # -------------------------------------------------
            # D. CREATE OUTPUT VIDEO
            # -------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as output_temp:

                output_path = output_temp.name

            output_width = 960
            output_height = 540

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (output_width, output_height)
            )

            if not writer.isOpened():
                raise RuntimeError(
                    "Could not initialize the output video writer."
                )

            # -------------------------------------------------
            # E. INITIALIZE COUNTERS
            # -------------------------------------------------

            counted_ids = set()
            previous_positions = {}

            vehicle_counts = {
                "car": 0,
                "motorcycle": 0,
                "bus": 0,
                "truck": 0
            }

            crossing_events = []

            # NEW: Store visible vehicle counts for each frame.
            vehicles_per_frame = []

            frames_processed = 0

            st.markdown(
                '<div class="vai-processing-note"><strong>Live processing</strong><br>'
                'VehicleAI is analyzing frames, tracking vehicle IDs, checking line crossings, '
                'and preparing the annotated video. Keep this tab open until processing finishes.</div>',
                unsafe_allow_html=True
            )
            progress_bar = st.progress(0, text="Preparing video analysis…")
            status_text = st.empty()
            preview_placeholder = st.empty()

            # -------------------------------------------------
            # F. PROCESS VIDEO FRAME BY FRAME
            # -------------------------------------------------

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.resize(
                    frame,
                    (output_width, output_height)
                )

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    imgsz=960,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                # Count visible vehicles in this frame.
                visible_vehicles = 0

                # Draw counting line.
                cv2.line(
                    frame,
                    (0, line_y),
                    (output_width, line_y),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "COUNTING LINE",
                    (10, max(30, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # ---------------------------------------------
                # DETECTION AND TRACKING
                # ---------------------------------------------

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):

                    boxes = (
                        result.boxes.xyxy.cpu().numpy()
                    )

                    class_ids = (
                        result.boxes.cls.cpu().numpy().astype(int)
                    )

                    track_ids = (
                        result.boxes.id.cpu().numpy().astype(int)
                    )

                    confidences = (
                        result.boxes.conf.cpu().numpy()
                    )

                    for box, class_id, track_id, box_conf in zip(
                        boxes,
                        class_ids,
                        track_ids,
                        confidences
                    ):

                        # Only process vehicles.
                        if class_id not in VEHICLE_CLASSES:
                            continue

                        vehicle_type = VEHICLE_CLASSES[class_id]

                        visible_vehicles += 1

                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # -------------------------------------
                        # LINE CROSSING COUNT
                        # -------------------------------------

                        if track_id not in counted_ids:

                            if track_id in previous_positions:

                                previous_y = previous_positions[
                                    track_id
                                ]

                                crossed_line = (
                                    (
                                        previous_y < line_y
                                        and center_y >= line_y
                                    )
                                    or
                                    (
                                        previous_y >= line_y
                                        and center_y < line_y
                                    )
                                )

                                if crossed_line:

                                    counted_ids.add(track_id)

                                    vehicle_counts[
                                        vehicle_type
                                    ] += 1

                                    crossing_events.append({
                                        "Track ID": int(track_id),
                                        "Vehicle Type": vehicle_type,
                                        "Timestamp (seconds)": round(
                                            frames_processed / fps,
                                            2
                                        ),
                                        "Direction": (
                                            "Downward"
                                            if center_y > previous_y
                                            else "Upward"
                                        )
                                    })

                            previous_positions[
                                track_id
                            ] = center_y

                        # -------------------------------------
                        # DRAW BOUNDING BOX
                        # -------------------------------------

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        label = (
                            f"{vehicle_type} "
                            f"ID:{track_id} "
                            f"{box_conf:.2f}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                # ---------------------------------------------
                # RECORD VEHICLE COUNT FOR THIS FRAME
                # ---------------------------------------------

                vehicles_per_frame.append(
                    visible_vehicles
                )

                # ---------------------------------------------
                # DRAW CURRENT FRAME DENSITY
                # ---------------------------------------------

                current_density = classify_traffic_density(
                    visible_vehicles,
                    medium_threshold,
                    high_threshold
                )

                if current_density == "Low":
                    density_color = (0, 255, 0)

                elif current_density == "Medium":
                    density_color = (0, 255, 255)

                else:
                    density_color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"Visible Vehicles: {visible_vehicles}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Traffic Density: {current_density}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    density_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Vehicles Crossed: {len(counted_ids)}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                # Write annotated frame.
                writer.write(frame)

                frames_processed += 1

                # ---------------------------------------------
                # UPDATE PROGRESS
                # ---------------------------------------------

                progress = min(
                    frames_processed / total_frames,
                    1.0
                )

                progress_bar.progress(progress)

                status_text.text(
                    f"Processing frame {frames_processed} "
                    f"of {total_frames}"
                )

                # Show occasional preview frames.
                preview_interval = max(
                    1,
                    int(fps * 2)
                )

                if (
                    frames_processed == 1
                    or frames_processed % preview_interval == 0
                ):

                    preview_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    preview_placeholder.image(
                        preview_frame,
                        caption=(
                            f"Processing preview: "
                            f"{frames_processed}/{total_frames}"
                        ),
                        width=720
                    )

            # -------------------------------------------------
            # G. RELEASE VIDEO RESOURCES
            # -------------------------------------------------

            cap.release()
            cap = None

            writer.release()
            writer = None

            progress_bar.progress(1.0)

            status_text.success(
                "Video processing completed!"
            )

            if frames_processed == 0:
                raise RuntimeError(
                    "No frames were processed."
                )

            # -------------------------------------------------
            # H. VALIDATE OUTPUT VIDEO
            # -------------------------------------------------

            if (
                not os.path.exists(output_path)
                or os.path.getsize(output_path) == 0
            ):
                raise RuntimeError(
                    "The annotated video is empty."
                )

            # -------------------------------------------------
            # I. CONVERT VIDEO FOR BROWSER PLAYBACK
            # -------------------------------------------------

            with st.spinner(
                "Preparing video for browser playback..."
            ):

                browser_video_path = convert_to_browser_video(
                    output_path
                )

                with open(
                    browser_video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()

            if not video_bytes:
                raise RuntimeError(
                    "Could not read the annotated video."
                )

            # Preserve video through Streamlit reruns.
            st.session_state[
                "annotated_video_bytes"
            ] = video_bytes

            # -------------------------------------------------
            # J. CALCULATE TRAFFIC DENSITY STATISTICS
            # -------------------------------------------------

            average_vehicles = (
                sum(vehicles_per_frame) / len(vehicles_per_frame)
            )

            peak_vehicles = max(
                vehicles_per_frame,
                default=0
            )

            overall_density = classify_traffic_density(
                average_vehicles,
                medium_threshold,
                high_threshold
            )

            # Frame-level time-series data.
            density_df = pd.DataFrame({
                "Frame": range(1, len(vehicles_per_frame) + 1),
                "Time (seconds)": [
                    round(i / fps, 2)
                    for i in range(len(vehicles_per_frame))
                ],
                "Visible Vehicles": vehicles_per_frame
            })

            # Classify each frame for density breakdown.
            density_df["Density"] = density_df[
                "Visible Vehicles"
            ].apply(
                lambda count: classify_traffic_density(
                    count,
                    medium_threshold,
                    high_threshold
                )
            )

            density_distribution = (
                density_df["Density"]
                .value_counts()
                .reindex(
                    ["Low", "Medium", "High"],
                    fill_value=0
                )
            )

            density_distribution_df = pd.DataFrame({
                "Traffic Density": density_distribution.index,
                "Frames": density_distribution.values
            })

            # -------------------------------------------------
            # K. PREPARE CROSSING EVENTS
            # -------------------------------------------------

            events_df = pd.DataFrame(
                crossing_events,
                columns=[
                    "Track ID",
                    "Vehicle Type",
                    "Timestamp (seconds)",
                    "Direction"
                ]
            )

            counts_df = pd.DataFrame({
                "Vehicle Type": list(vehicle_counts.keys()),
                "Count": list(vehicle_counts.values())
            })

            # Group crossings into 2-second intervals.
            if not events_df.empty:

                events_df["Time Interval"] = (
                    (
                        events_df["Timestamp (seconds)"] // 2
                    ) * 2
                ).astype(int)

                interval_df = (
                    events_df.groupby(
                        ["Time Interval", "Vehicle Type"]
                    )
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )

            else:

                interval_df = pd.DataFrame()

            # Calculate observed crossing rate.
            observed_duration = frames_processed / fps

            crossing_rate = (
                len(counted_ids) / observed_duration * 60
                if observed_duration > 0
                else 0
            )

            # -------------------------------------------------
            # L. SAVE RESULTS IN SESSION STATE
            # -------------------------------------------------

            st.session_state["traffic_results"] = {
                "vehicle_counts": vehicle_counts,
                "total_crossings": len(counted_ids),
                "duration": observed_duration,
                "crossing_rate": crossing_rate,
                "counts_df": counts_df,
                "events_df": events_df,
                "interval_df": interval_df,
                "average_vehicles": average_vehicles,
                "peak_vehicles": peak_vehicles,
                "overall_density": overall_density,
                "density_df": density_df,
                "density_distribution_df": density_distribution_df,
                "medium_threshold": medium_threshold,
                "high_threshold": high_threshold,
                "frames_processed": frames_processed
            }

            st.success("Analysis finished successfully!")

        except Exception as error:

            st.error(
                f"An error occurred during analysis: {error}"
            )

            st.exception(error)

        finally:

            if cap is not None:
                cap.release()

            if writer is not None:
                writer.release()

            # Remove temporary files after their contents
            # have been read into memory.
            temp_paths = {
                input_path,
                output_path,
                browser_video_path
            }

            for temp_path in temp_paths:

                if temp_path and os.path.exists(temp_path):

                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass


# =========================================================
# 10. DISPLAY RESULTS
# =========================================================

results_data = st.session_state["traffic_results"]

if results_data is not None:

    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vai-overview-label">Dashboard overview</div>', unsafe_allow_html=True)
    st.subheader("Your traffic snapshot")

    overview_cols = st.columns(4)
    overview_cols[0].metric("🚘 Total crossings", results_data["total_crossings"])
    overview_cols[1].metric("🚗 Cars", results_data["vehicle_counts"]["car"])
    overview_cols[2].metric(
        "🏍️ Motorcycles", results_data["vehicle_counts"]["motorcycle"]
    )
    overview_cols[3].metric(
        "🚦 Traffic density", results_data["overall_density"]
    )
    st.caption(
        "Summary of this processed clip. Counts and density are estimates from the selected video."
    )

    st.divider()
    st.markdown('<div id="results"></div>', unsafe_allow_html=True)
    st.header("📊 Traffic Analysis Results")

    vehicle_counts = results_data["vehicle_counts"]

    # -----------------------------------------------------
    # VEHICLE COUNT METRICS
    # -----------------------------------------------------

    st.subheader("🚗 Vehicle Crossings")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Crossings",
        results_data["total_crossings"]
    )

    col2.metric(
        "Cars",
        vehicle_counts["car"]
    )

    col3.metric(
        "Motorcycles",
        vehicle_counts["motorcycle"]
    )

    col4.metric(
        "Buses + Trucks",
        vehicle_counts["bus"] + vehicle_counts["truck"]
    )

    col5, col6 = st.columns(2)

    col5.metric(
        "Video Duration",
        f"{results_data['duration']:.2f} seconds"
    )

    col6.metric(
        "Observed Crossings / Minute",
        f"{results_data['crossing_rate']:.1f}"
    )

    st.caption(
        "The crossing rate describes the analyzed video. "
        "It should not be extrapolated into a reliable hourly "
        "estimate from a short recording."
    )

    # -----------------------------------------------------
    # NEW: TRAFFIC DENSITY CLASSIFICATION
    # -----------------------------------------------------

    st.divider()

    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.header("🚦 Traffic Density Analysis")

    average_vehicles = results_data["average_vehicles"]
    peak_vehicles = results_data["peak_vehicles"]
    overall_density = results_data["overall_density"]

    density_col1, density_col2, density_col3 = st.columns(3)

    density_col1.metric(
        "Average Visible Vehicles",
        f"{average_vehicles:.2f}"
    )

    density_col2.metric(
        "Peak Visible Vehicles",
        peak_vehicles
    )

    density_col3.metric(
        "Overall Traffic Density",
        overall_density
    )

    density_descriptions = {
        "Low": "🟢 Low traffic: relatively few vehicles are visible.",
        "Medium": "🟡 Medium traffic: a moderate number of vehicles are visible.",
        "High": "🔴 High traffic: many vehicles are visible simultaneously."
    }

    st.info(
        density_descriptions[overall_density]
    )

    st.write(
        f"**Classification rules:** Low: average below "
        f"{results_data['medium_threshold']}; Medium: average "
        f"from {results_data['medium_threshold']} to below "
        f"{results_data['high_threshold']}; High: average at or "
        f"above {results_data['high_threshold']} vehicles per frame."
    )

    st.caption(
        "These thresholds are prototype settings. Traffic density "
        "is estimated from visible vehicle counts, not measured "
        "road occupancy or a validated traffic-engineering model."
    )

    # -----------------------------------------------------
    st.markdown('<div id="annotated-video"></div>', unsafe_allow_html=True)
    st.header("🎥 Annotated Video")

    video_bytes = st.session_state["annotated_video_bytes"]

    if video_bytes:

        st.success(
            "Annotated video is ready."
        )

        st.video(
            video_bytes,
            format="video/mp4"
        )

        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="vehicleai_annotated.mp4",
            mime="video/mp4",
            key="download_annotated_video"
        )

    else:

        st.warning(
            "The annotated video is not available."
        )


    # ANALYTICS DASHBOARD — TWO-COLUMN GRID

    st.markdown('<div class="vai-section-label">Analytics workspace</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        st.subheader("Vehicle Crossings by Type")
        counts_df = results_data["counts_df"]
        st.bar_chart(
            counts_df.set_index("Vehicle Type")["Count"],
            horizontal=True,
            use_container_width=True
        )

        st.subheader("Traffic Density Distribution")
        density_distribution_df = results_data["density_distribution_df"]
        st.bar_chart(
            density_distribution_df.set_index("Traffic Density")["Frames"],
            use_container_width=True
        )
        st.caption("Number of processed frames in each density category.")

    with chart_right:
        st.subheader("Visible Vehicles Over Time")
        density_df = results_data["density_df"]
        density_chart_df = density_df[
            ["Time (seconds)", "Visible Vehicles"]
        ].set_index("Time (seconds)")
        st.line_chart(
            density_chart_df,
            use_container_width=True
        )

        st.subheader("Vehicle Crossings Every 2 Seconds")
        interval_df = results_data["interval_df"]
        if not interval_df.empty:
            st.bar_chart(interval_df, use_container_width=True)
        else:
            st.info(
                "No vehicles crossed the counting line, so there is "
                "no interval chart to display."
            )

    # CROSSING EVENTS TABLE
    # -----------------------------------------------------

    st.subheader("📋 Vehicle Crossing Events")

    events_df = results_data["events_df"]

    if not events_df.empty:

        st.dataframe(
            events_df,
            use_container_width=True
        )

    else:

        st.info(
            "No vehicles crossed the counting line."
        )

    # -----------------------------------------------------
    # DOWNLOAD CSV
    # -----------------------------------------------------

    st.download_button(
        label="📄 Download Crossing Events CSV",
        data=events_df.to_csv(index=False).encode("utf-8"),
        file_name="vehicleai_crossing_events.csv",
        mime="text/csv",
        key="download_crossing_events"
    )

    # Download frame-level density data.
    st.download_button(
        label="📊 Download Traffic Density CSV",
        data=density_df.to_csv(index=False).encode("utf-8"),
        file_name="vehicleai_traffic_density.csv",
        mime="text/csv",
        key="download_density_csv"
    )

    # -----------------------------------------------------
    # ANNOTATED VIDEO
    # -----------------------------------------------------

    st.divider()



# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "VehicleAI | YOLO26 | ByteTrack | Traffic Analytics"
)

st.caption(
    "Prototype note: Detection, tracking, line crossings, and "
    "traffic density can be affected by occlusion, missed "
    "detections, tracking errors, video perspective, and "
    "threshold selection."
)
