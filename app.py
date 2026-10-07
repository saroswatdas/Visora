import html
import time
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules.preprocessing import preprocess_image
from modules.ocr import extract_text_with_confidence
from modules.object_detection import detect_objects


st.set_page_config(
    page_title="Visora — Computer Vision",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HELPERS
# ============================================================

def file_key(uploaded):
    """Identify an uploaded file so stale results can be hidden."""
    return f"{uploaded.name}-{uploaded.size}"


def load_image(uploaded):
    image = Image.open(uploaded).convert("RGB")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image, cv_image


def format_confidence(value):
    """SSD scores saturate near 1.0, so avoid showing a misleading 100.0%."""
    pct = value * 100
    return ">99.9%" if pct >= 99.95 else f"{pct:.1f}%"


def to_png_bytes(image_bgr):
    ok, buffer = cv2.imencode(".png", image_bgr)
    return buffer.tobytes() if ok else b""


def card_header(title, subtitle):
    st.html(
        f"""
        <div class="card-head">
            <div class="result-title">{html.escape(title)}</div>
            <div class="result-subtitle">{html.escape(subtitle)}</div>
        </div>
        """
    )


def chips(items):
    inner = "".join(f'<span class="chip">{html.escape(i)}</span>' for i in items)
    return f'<div class="chip-row">{inner}</div>'


def placeholder(message):
    st.html(f'<div class="placeholder">{html.escape(message)}</div>')


def empty_state(title, hint):
    st.html(
        f"""
        <div class="empty-state">
            <div class="empty-title">{html.escape(title)}</div>
            <div class="empty-hint">{html.escape(hint)}</div>
        </div>
        """
    )


def detection_list_html(detections):
    rows = []
    for d in detections:
        name = html.escape(d.get("object", "Unknown"))
        conf = d.get("confidence", 0)
        width = max(2, min(100, conf * 100))
        score = html.escape(format_confidence(conf))
        rows.append(
            f"""
            <div class="det-row">
                <span class="det-name">{name}</span>
                <div class="det-bar"><div class="det-fill" style="width:{width:.1f}%"></div></div>
                <span class="det-score">{score}</span>
            </div>
            """
        )
    return f'<div class="det-list">{"".join(rows)}</div>'


PREPROCESS_MODES = {
    "clean": "Clean — grayscale + resize (recommended)",
    "otsu": "Otsu — high-contrast binarization",
    "adaptive": "Adaptive — uneven lighting / photos",
}

PSM_MODES = {
    3: "PSM 3 — Automatic layout",
    6: "PSM 6 — Uniform text block",
    7: "PSM 7 — Single text line",
    11: "PSM 11 — Sparse text",
}


# ============================================================
# VISORA UI
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 12% 12%, rgba(91,72,255,.10), transparent 30%),
        radial-gradient(circle at 88% 22%, rgba(64,174,255,.07), transparent 28%),
        #07090d;
    color: #f5f7fb;
    font-family: 'Inter', sans-serif;
}

/* Main container: the selectors below cover older and newer Streamlit
   versions. !important is needed to beat Streamlit's default top padding. */
.main .block-container,
.stMainBlockContainer,
[data-testid="stMainBlockContainer"] {
    max-width: 1180px !important;
    padding: 14px 34px 70px !important;
}

#MainMenu,
footer,
[data-testid="stDecoration"],
.stAppDeployButton {
    display: none !important;
}

header {
    background: transparent !important;
}

/* ---------- TOP BAR ---------- */

.topbar {
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,.07);
    margin-bottom: 26px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-mark {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#735cff,#4d8cff);
    color: white;
    font-size: 17px;
    font-weight: 800;
    box-shadow: 0 8px 25px rgba(100,85,255,.28);
}

.brand-name {
    color: #f7f8fb;
    font-size: 22px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -.4px;
}

.brand-subtitle {
    color: #8791a6;
    font-size: 12px;
    line-height: 1;
    margin-left: 9px;
    position: relative;
    top: 1px;
}

.online {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #62e3a2;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #42dfa0;
    box-shadow: 0 0 10px rgba(66,223,160,.7);
}

/* ---------- HERO ---------- */

.hero-grid {
    display: grid;
    grid-template-columns: 1.2fr .8fr;
    gap: 56px;
    align-items: start;
    margin: 0 0 20px;
}

.eyebrow {
    color: #8174ff;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 17px;
}

.hero-title {
    color: #fafbff;
    font-size: clamp(44px,4.8vw,60px);
    line-height: 1;
    letter-spacing: -3px;
    font-weight: 800;
}

.hero-gradient {
    background: linear-gradient(100deg,#a99aff,#66b9ff 55%,#54d8e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-copy {
    max-width: 550px;
    margin-top: 23px;
    color: #9aa3b8;
    font-size: 15px;
    line-height: 1.75;
}

.hero-stats {
    display: flex;
    gap: 26px;
    margin-top: 29px;
}

.hero-stat {
    padding-left: 14px;
    border-left: 1px solid rgba(255,255,255,.09);
}

.hero-stat:first-child {
    padding-left: 0;
    border-left: 0;
}

.hero-stat strong {
    display: block;
    color: #f1f3f8;
    font-size: 14px;
}

.hero-stat span {
    display: block;
    color: #7d879c;
    font-size: 10px;
    margin-top: 4px;
    letter-spacing: .5px;
    text-transform: uppercase;
}

/* ---------- WORKSPACE CARD ---------- */

.workspace-card {
    min-height: 250px;
    margin-top: 6px;
    padding: 25px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.09);
    background: linear-gradient(145deg,rgba(20,24,36,.97),rgba(12,15,23,.97));
    box-shadow: 0 25px 70px rgba(0,0,0,.30);
}

.workspace-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
}

.workspace-title {
    color: #f4f6fb;
    font-size: 18px;
    font-weight: 700;
}

.workspace-subtitle {
    color: #8791a6;
    font-size: 12px;
    margin-top: 5px;
}

.engine-badge {
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid rgba(111,96,255,.25);
    background: rgba(99,90,255,.09);
    color: #a79dff;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.engine-list {
    margin-top: 22px;
    display: grid;
    gap: 10px;
}

.engine-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 13px 14px;
    border-radius: 13px;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.025);
}

.engine-icon {
    flex: 0 0 34px;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(103,91,255,.14);
    color: #a79dff;
    font-size: 13px;
    font-weight: 800;
}

.engine-name {
    color: #eef0f6;
    font-size: 13px;
    font-weight: 700;
}

.engine-desc {
    color: #8791a6;
    font-size: 11px;
    margin-top: 3px;
}

/* ---------- TABS ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,.07);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 17px;
    color: #8791a6;
    font-size: 13px;
    font-weight: 650;
    border-radius: 9px 9px 0 0;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: rgba(103,91,255,.10);
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #7565ff;
}

/* ---------- UPLOAD (compact) ---------- */

[data-testid="stFileUploader"] {
    margin-top: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    background:
        radial-gradient(circle at center,rgba(101,88,255,.07),transparent 55%),
        #0d111a;
    border: 1px dashed rgba(128,137,164,.30);
    border-radius: 15px;
    min-height: 96px;
    padding: 14px 18px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(126,111,255,.75);
}

/* ---------- CAMERA ---------- */

[data-testid="stCameraInput"] {
    margin-top: 10px;
}

[data-testid="stCameraInput"] > div,
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 15px;
}

/* ---------- BUTTONS ---------- */

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 43px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.10);
    background: #151a25;
    color: #f1f3f8;
    font-weight: 650;
    font-size: 13px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(119,103,255,.75);
    color: white;
    background: #1a1f2d;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6e5cff,#4d8cff);
    border: 0;
    color: white;
}

/* ---------- RESULT UI ---------- */

.section-label {
    color: #8791a6;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 23px 0 8px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
    border-color: rgba(255,255,255,.08) !important;
    background: #0d1119;
}

.card-head {
    margin-bottom: 4px;
}

.result-title {
    color: #f1f3f8;
    font-size: 14px;
    font-weight: 700;
}

.result-subtitle {
    color: #8791a6;
    font-size: 11px;
    margin-top: 4px;
}

.placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 220px;
    border: 1px dashed rgba(128,137,164,.25);
    border-radius: 12px;
    color: #6f7a90;
    font-size: 12px;
    text-align: center;
    padding: 20px;
}

.empty-state {
    margin-top: 14px;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.06);
    background: rgba(255,255,255,.02);
}

.empty-title {
    color: #e6e9f2;
    font-size: 13px;
    font-weight: 650;
}

.empty-hint {
    color: #8791a6;
    font-size: 12px;
    margin-top: 5px;
    line-height: 1.6;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 4px 0 12px;
}

.chip {
    padding: 5px 11px;
    border-radius: 999px;
    border: 1px solid rgba(111,96,255,.25);
    background: rgba(99,90,255,.09);
    color: #b4abff;
    font-size: 11px;
    font-weight: 650;
}

/* ---------- COMPACT DETECTION LIST ---------- */

.det-list {
    display: grid;
    gap: 8px;
}

.det-row {
    display: grid;
    grid-template-columns: 110px 1fr 64px;
    align-items: center;
    gap: 14px;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.06);
    background: #0a0e15;
}

.det-name {
    color: #eef0f6;
    font-size: 13px;
    font-weight: 650;
    text-transform: capitalize;
}

.det-bar {
    height: 6px;
    border-radius: 999px;
    background: rgba(255,255,255,.07);
    overflow: hidden;
}

.det-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#7565ff,#54d8e8);
}

.det-score {
    color: #dce1eb;
    font-size: 12px;
    font-weight: 700;
    text-align: right;
}

.stTextArea textarea {
    background: #0a0e15 !important;
    color: #dce1eb !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 11px !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

[data-testid="stAlert"] {
    border-radius: 11px;
}

.footer {
    text-align: center;
    color: #5a6479;
    font-size: 11px;
    margin-top: 55px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,.05);
}

@media (max-width:850px) {
    .hero-grid {
        grid-template-columns: 1fr;
        gap: 30px;
    }

    .hero-title {
        font-size: 46px;
    }

    .brand-subtitle {
        display: none;
    }

    .main .block-container,
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"] {
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    .det-row {
        grid-template-columns: 90px 1fr 58px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


st.html(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div>
                <span class="brand-name">Visora</span>
                <span class="brand-subtitle">Computer Vision</span>
            </div>
        </div>
        <div class="online">
            <span class="online-dot"></span>
            System Online
        </div>
    </div>
    """
)


st.html(
    """
    <div class="hero-grid">
        <div>
            <div class="eyebrow">Computer Vision Workspace</div>

            <div class="hero-title">
                See the image.<br>
                <span class="hero-gradient">Understand the data.</span>
            </div>

            <div class="hero-copy">
                Transform visual data into machine-readable intelligence.
                Extract text with Tesseract OCR or identify supported objects
                using a pre-trained MobileNet-SSD model.
            </div>

            <div class="hero-stats">
                <div class="hero-stat">
                    <strong>Tesseract</strong>
                    <span>Text recognition</span>
                </div>

                <div class="hero-stat">
                    <strong>MobileNet-SSD</strong>
                    <span>Object detection</span>
                </div>

                <div class="hero-stat">
                    <strong>80%+</strong>
                    <span>Validated confidence</span>
                </div>
            </div>
        </div>

        <div class="workspace-card">
            <div class="workspace-head">
                <div>
                    <div class="workspace-title">Analyze an image</div>
                    <div class="workspace-subtitle">
                        Choose an engine and upload visual data.
                    </div>
                </div>

                <div class="engine-badge">AI Engine</div>
            </div>

            <div class="engine-list">
                <div class="engine-row">
                    <div class="engine-icon">Aa</div>
                    <div>
                        <div class="engine-name">Text Recognition</div>
                        <div class="engine-desc">
                            Tesseract OCR with selectable preprocessing
                            and layout modes.
                        </div>
                    </div>
                </div>

                <div class="engine-row">
                    <div class="engine-icon">Ob</div>
                    <div>
                        <div class="engine-name">Object Detection</div>
                        <div class="engine-desc">
                            MobileNet-SSD with an adjustable confidence
                            threshold.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
)


tab_ocr, tab_detection, tab_camera = st.tabs(
    ["Text Recognition", "Object Detection", "Camera"]
)


# ============================================================
# OCR
# ============================================================

with tab_ocr:

    st.html(
        """
        <div class="section-label">
            Tesseract OCR · Text extraction
        </div>
        """
    )

    uploaded_ocr = st.file_uploader(
        "Upload an image to analyze",
        type=["jpg", "jpeg", "png"],
        key="ocr_upload",
        help="JPG, JPEG and PNG images are supported.",
    )

    if uploaded_ocr is None:

        empty_state(
            "No image yet",
            "Upload a screenshot, scan, or photo of a document. "
            "Clean, high-contrast text gives the best results.",
        )

    else:

        image, cv_image = load_image(uploaded_ocr)

        st.html('<div class="section-label">OCR configuration</div>')

        cfg_col1, cfg_col2, cfg_col3 = st.columns(
            [3, 3, 1.4], gap="large", vertical_alignment="bottom"
        )

        with cfg_col1:
            psm_mode = st.selectbox(
                "Page segmentation mode",
                list(PSM_MODES.keys()),
                format_func=lambda x: PSM_MODES[x],
                key="psm_mode",
            )

        with cfg_col2:
            prep_mode = st.selectbox(
                "Preprocessing",
                list(PREPROCESS_MODES.keys()),
                format_func=lambda x: PREPROCESS_MODES[x],
                key="prep_mode",
            )

        with cfg_col3:
            run_ocr = st.button(
                "Run OCR",
                key="run_ocr",
                type="primary",
            )

        if run_ocr:

            with st.spinner("Analyzing image with Tesseract..."):
                start = time.perf_counter()
                processed = preprocess_image(cv_image, mode=prep_mode)
                text, mean_conf = extract_text_with_confidence(
                    processed, psm_mode
                )
                elapsed = time.perf_counter() - start

            st.session_state["ocr_result"] = {
                "file": file_key(uploaded_ocr),
                "processed": processed,
                "text": text,
                "confidence": mean_conf,
                "elapsed": elapsed,
            }

        result = st.session_state.get("ocr_result")
        has_result = bool(
            result and result["file"] == file_key(uploaded_ocr)
        )

        left, right = st.columns(2, gap="large")

        with left:

            with st.container(border=True):

                card_header("Input image", "Original upload.")
                st.image(image, use_container_width=True)

                if has_result:
                    with st.expander("View processed image"):
                        st.image(
                            result["processed"],
                            channels="GRAY",
                            use_container_width=True,
                        )

        with right:

            with st.container(border=True):

                card_header(
                    "Recognized text",
                    "Machine-readable OCR output.",
                )

                if not has_result:

                    placeholder("Run OCR to see the recognized text here.")

                elif result["text"]:

                    word_count = len(result["text"].split())

                    st.html(
                        chips(
                            [
                                f"{result['confidence']:.1f}% mean confidence",
                                f"{word_count} words",
                                f"{result['elapsed']:.2f} s",
                            ]
                        )
                    )

                    st.text_area(
                        "OCR output",
                        result["text"],
                        height=360,
                        label_visibility="collapsed",
                    )

                    st.download_button(
                        "Download text (.txt)",
                        data=result["text"],
                        file_name="visora_ocr.txt",
                        mime="text/plain",
                        key="dl_ocr_text",
                    )

                else:

                    st.info("No readable text was detected.")


# ============================================================
# OBJECT DETECTION
# ============================================================

with tab_detection:

    st.html(
        """
        <div class="section-label">
            MobileNet-SSD · Object detection
        </div>
        """
    )

    uploaded_detection = st.file_uploader(
        "Upload an image to analyze",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
        help="JPG, JPEG and PNG images are supported.",
    )

    if uploaded_detection is None:

        empty_state(
            "No image yet",
            "Try a photo with cars, people, buses, bicycles, chairs or "
            "bottles. The model recognizes 20 everyday object classes.",
        )

    else:

        image, cv_image = load_image(uploaded_detection)

        ctrl_col, btn_col = st.columns(
            [4, 1.2], gap="large", vertical_alignment="bottom"
        )

        with ctrl_col:
            threshold_pct = st.slider(
                "Confidence threshold",
                min_value=30,
                max_value=95,
                value=80,
                step=5,
                format="%d%%",
                key="conf_threshold",
                help="Lower values find more objects but with more "
                     "false positives.",
            )

        with btn_col:
            run_detection = st.button(
                "Run detection",
                key="run_detection",
                type="primary",
            )

        st.caption(
            f"Only detections with confidence ≥ {threshold_pct}% "
            "are validated."
        )

        if run_detection:

            with st.spinner("Running MobileNet-SSD inference..."):
                start = time.perf_counter()
                annotated_image, detections = detect_objects(
                    cv_image,
                    conf_threshold=threshold_pct / 100,
                )
                elapsed = time.perf_counter() - start

            st.session_state["det_result"] = {
                "file": file_key(uploaded_detection),
                "annotated": annotated_image,
                "detections": detections,
                "threshold": threshold_pct,
                "elapsed": elapsed,
            }

        result = st.session_state.get("det_result")
        has_result = bool(
            result and result["file"] == file_key(uploaded_detection)
        )

        left, right = st.columns(2, gap="large")

        with left:

            with st.container(border=True):

                card_header("Input image", "Original upload.")
                st.image(image, use_container_width=True)

        with right:

            with st.container(border=True):

                card_header(
                    "Detection result",
                    "Validated detections with bounding boxes.",
                )

                if has_result:

                    st.image(
                        cv2.cvtColor(
                            result["annotated"], cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download annotated image (.png)",
                        data=to_png_bytes(result["annotated"]),
                        file_name="visora_detection.png",
                        mime="image/png",
                        key="dl_det_image",
                    )

                else:

                    placeholder(
                        "Run detection to see bounding boxes here."
                    )

        if has_result:

            detections = result["detections"]

            with st.container(border=True):

                card_header(
                    "Detected objects",
                    "Confidence scores from MobileNet-SSD.",
                )

                if detections:

                    counts = Counter(d["object"] for d in detections)

                    st.html(
                        chips(
                            [
                                f"{len(detections)} object(s)",
                                f"{len(counts)} class(es)",
                                f"{result['elapsed']:.2f} s",
                                f"threshold ≥ {result['threshold']}%",
                            ]
                        )
                    )

                    st.html(detection_list_html(detections))

                else:

                    st.info(
                        f"No objects detected with confidence "
                        f"≥ {result['threshold']}%. Try lowering the "
                        "threshold."
                    )


# ============================================================
# CAMERA
# ============================================================

with tab_camera:

    st.html(
        """
        <div class="section-label">
            Camera · Capture and analyze
        </div>
        """
    )

    cam_engine = st.radio(
        "Engine",
        ["Object detection", "Text recognition"],
        horizontal=True,
        key="cam_engine",
    )

    # The camera widget is only created after the toggle is switched on,
    # so the browser asks for camera permission at that moment and not
    # when the page first loads.
    cam_on = st.toggle(
        "Turn on camera",
        key="cam_on",
        help="The browser will ask for camera permission when you switch "
             "this on. Camera access works on localhost or HTTPS.",
    )

    photo = None

    if cam_on:
        photo = st.camera_input(
            "Take a photo",
            key="cam_photo",
        )

    if not cam_on:

        empty_state(
            "Camera is off",
            "Switch on “Turn on camera” above. Your browser will then ask "
            "for permission to use the camera.",
        )

    elif photo is None:

        empty_state(
            "No photo yet",
            "Take a photo with the camera above. Use object detection "
            "for scenes, or text recognition for a page held up to the "
            "camera.",
        )

    elif cam_engine == "Object detection":

        image, cv_image = load_image(photo)

        ctrl_col, btn_col = st.columns(
            [4, 1.2], gap="large", vertical_alignment="bottom"
        )

        with ctrl_col:
            cam_threshold = st.slider(
                "Confidence threshold",
                min_value=30,
                max_value=95,
                value=80,
                step=5,
                format="%d%%",
                key="cam_conf_threshold",
                help="Lower values find more objects but with more "
                     "false positives.",
            )

        with btn_col:
            run_cam_detection = st.button(
                "Run detection",
                key="run_cam_detection",
                type="primary",
            )

        st.caption(
            f"Only detections with confidence ≥ {cam_threshold}% "
            "are validated."
        )

        if run_cam_detection:

            with st.spinner("Running MobileNet-SSD inference..."):
                start = time.perf_counter()
                annotated_image, detections = detect_objects(
                    cv_image,
                    conf_threshold=cam_threshold / 100,
                )
                elapsed = time.perf_counter() - start

            st.session_state["cam_det_result"] = {
                "file": file_key(photo),
                "annotated": annotated_image,
                "detections": detections,
                "threshold": cam_threshold,
                "elapsed": elapsed,
            }

        result = st.session_state.get("cam_det_result")
        has_result = bool(result and result["file"] == file_key(photo))

        left, right = st.columns(2, gap="large")

        with left:

            with st.container(border=True):

                card_header("Captured photo", "Original capture.")
                st.image(image, use_container_width=True)

        with right:

            with st.container(border=True):

                card_header(
                    "Detection result",
                    "Validated detections with bounding boxes.",
                )

                if has_result:

                    st.image(
                        cv2.cvtColor(
                            result["annotated"], cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download annotated image (.png)",
                        data=to_png_bytes(result["annotated"]),
                        file_name="visora_camera_detection.png",
                        mime="image/png",
                        key="dl_cam_det_image",
                    )

                else:

                    placeholder(
                        "Run detection to see bounding boxes here."
                    )

        if has_result:

            detections = result["detections"]

            with st.container(border=True):

                card_header(
                    "Detected objects",
                    "Confidence scores from MobileNet-SSD.",
                )

                if detections:

                    counts = Counter(d["object"] for d in detections)

                    st.html(
                        chips(
                            [
                                f"{len(detections)} object(s)",
                                f"{len(counts)} class(es)",
                                f"{result['elapsed']:.2f} s",
                                f"threshold ≥ {result['threshold']}%",
                            ]
                        )
                    )

                    st.html(detection_list_html(detections))

                else:

                    st.info(
                        f"No objects detected with confidence "
                        f"≥ {result['threshold']}%. Try lowering the "
                        "threshold."
                    )

    else:

        image, cv_image = load_image(photo)

        cam_cfg1, cam_cfg2, cam_cfg3 = st.columns(
            [3, 3, 1.4], gap="large", vertical_alignment="bottom"
        )

        with cam_cfg1:
            cam_psm = st.selectbox(
                "Page segmentation mode",
                list(PSM_MODES.keys()),
                format_func=lambda x: PSM_MODES[x],
                key="cam_psm",
            )

        with cam_cfg2:
            # Adaptive is the better default for photos of paper
            cam_prep = st.selectbox(
                "Preprocessing",
                list(PREPROCESS_MODES.keys()),
                index=list(PREPROCESS_MODES.keys()).index("adaptive"),
                format_func=lambda x: PREPROCESS_MODES[x],
                key="cam_prep",
            )

        with cam_cfg3:
            run_cam_ocr = st.button(
                "Run OCR",
                key="run_cam_ocr",
                type="primary",
            )

        if run_cam_ocr:

            with st.spinner("Analyzing photo with Tesseract..."):
                start = time.perf_counter()
                processed = preprocess_image(cv_image, mode=cam_prep)
                text, mean_conf = extract_text_with_confidence(
                    processed, cam_psm
                )
                elapsed = time.perf_counter() - start

            st.session_state["cam_ocr_result"] = {
                "file": file_key(photo),
                "processed": processed,
                "text": text,
                "confidence": mean_conf,
                "elapsed": elapsed,
            }

        result = st.session_state.get("cam_ocr_result")
        has_result = bool(result and result["file"] == file_key(photo))

        left, right = st.columns(2, gap="large")

        with left:

            with st.container(border=True):

                card_header("Captured photo", "Original capture.")
                st.image(image, use_container_width=True)

                if has_result:
                    with st.expander("View processed image"):
                        st.image(
                            result["processed"],
                            channels="GRAY",
                            use_container_width=True,
                        )

        with right:

            with st.container(border=True):

                card_header(
                    "Recognized text",
                    "Machine-readable OCR output.",
                )

                if not has_result:

                    placeholder("Run OCR to see the recognized text here.")

                elif result["text"]:

                    word_count = len(result["text"].split())

                    st.html(
                        chips(
                            [
                                f"{result['confidence']:.1f}% mean confidence",
                                f"{word_count} words",
                                f"{result['elapsed']:.2f} s",
                            ]
                        )
                    )

                    st.text_area(
                        "OCR output",
                        result["text"],
                        height=360,
                        label_visibility="collapsed",
                        key="cam_ocr_text",
                    )

                    st.download_button(
                        "Download text (.txt)",
                        data=result["text"],
                        file_name="visora_camera_ocr.txt",
                        mime="text/plain",
                        key="dl_cam_ocr_text",
                    )

                else:

                    st.info("No readable text was detected.")


st.html(
    """
    <div class="footer">
        Visora · Computer Vision
    </div>
    """
)