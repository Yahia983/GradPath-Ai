import streamlit as st
import streamlit.components.v1 as components
import json
import os
import datetime
import random
import hashlib
import math
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GradPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  dark academic / editorial look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: #0d0f14;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #2a2d38;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stButton button {
    color: #c8c4bc !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    letter-spacing: -0.02em;
}

/* Cards / panels */
.grad-card {
    background: #181c26;
    border: 1px solid #252834;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
}

.grad-card-accent {
    background: linear-gradient(135deg, #1a1f2e 0%, #151922 100%);
    border: 1px solid #2e3650;
    border-left: 3px solid #c9a84c;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stTextArea textarea {
    background: #1e2230 !important;
    border: 1px solid #2e3248 !important;
    color: #e8e4dc !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #c9a84c !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #e0bf6a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
}

/* Sidebar buttons – ghost style */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #a0a8c0 !important;
    border: 1px solid #2a2d38 !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 0.75rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
    background: transparent !important;
    transform: none;
    box-shadow: none !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: #181c26 !important;
    border-radius: 12px !important;
    border: 1px solid #252834 !important;
    margin-bottom: 0.6rem !important;
}

/* Sliders */
.stSlider .st-bf { background: #c9a84c !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #181c26 !important;
    border: 1px solid #252834 !important;
    border-radius: 8px !important;
    color: #e8e4dc !important;
    font-family: 'Playfair Display', serif !important;
}

/* Progress / metric */
.stMetric { background: #181c26; border-radius: 10px; padding: 0.8rem 1rem; }

/* Divider */
hr { border-color: #252834 !important; }

/* Gold accent text */
.gold { color: #c9a84c; }
.muted { color: #6b7080; font-size: 0.85rem; }

/* Score badge */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #c9a84c, #e0bf6a);
    color: #0d0f14;
    font-weight: 700;
    font-size: 2rem;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-family: 'Playfair Display', serif;
}

/* Probability bar wrapper */
.prob-bar-wrap {
    background: #252834;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 0.4rem;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #c9a84c, #e0bf6a);
    transition: width 0.6s ease;
}

/* Warning / info boxes */
.info-box {
    background: #1a1f2e;
    border-left: 3px solid #4a90d9;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #a8b8d8;
    margin: 0.6rem 0;
}
.warn-box {
    background: #1e1a10;
    border-left: 3px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #c8b87a;
    margin: 0.6rem 0;
}

/* ─────────────────────────────────────────────
   3D SCENERY ENHANCEMENTS
   - Let the WebGL starfield/constellation behind
     the app show through translucent panels
   - Add real 3D depth (perspective tilt + glass)
     to existing cards without changing markup
───────────────────────────────────────────── */

/* Allow the fixed 3D canvas (injected into the page body) to show through */
.stApp {
    background: rgba(13, 15, 20, 0.78) !important;
}
[data-testid="stSidebar"] {
    background: rgba(19, 22, 30, 0.82) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stHeader"] {
    background: rgba(13, 15, 20, 0) !important;
}

/* Glass + 3D tilt for cards */
.grad-card,
.grad-card-accent,
[data-testid="stChatMessage"],
.streamlit-expanderHeader,
.stMetric {
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transform-style: preserve-3d;
    transition: transform 0.35s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s ease, border-color 0.35s ease;
    will-change: transform;
}
.grad-card:hover,
.grad-card-accent:hover {
    transform: perspective(900px) rotateX(2.5deg) rotateY(-2.5deg) translateY(-5px) translateZ(6px);
    box-shadow: 0 22px 45px rgba(0,0,0,0.45), 0 0 28px rgba(201,168,76,0.10);
    border-color: #3a3f55;
}

/* Score badge — 3D pop-in + gentle floating glow */
@keyframes grad-badge-pop {
    0%   { transform: scale(0.6) rotateY(75deg); opacity: 0; }
    60%  { transform: scale(1.05) rotateY(-8deg); opacity: 1; }
    100% { transform: scale(1) rotateY(0deg); opacity: 1; }
}
@keyframes grad-badge-float {
    0%, 100% { transform: translateY(0) rotateZ(0deg); box-shadow: 0 8px 24px rgba(201,168,76,0.25); }
    50%      { transform: translateY(-4px) rotateZ(0.6deg); box-shadow: 0 16px 36px rgba(201,168,76,0.38); }
}
.score-badge {
    animation: grad-badge-pop 0.7s cubic-bezier(.2,.8,.2,1) both,
               grad-badge-float 4.5s ease-in-out 0.7s infinite;
    transform-style: preserve-3d;
}

/* Subtle gold "stardust" drift across info/warn boxes */
@keyframes grad-shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
.warn-box, .info-box {
    background-image: linear-gradient(120deg, rgba(201,168,76,0.0) 0%, rgba(201,168,76,0.08) 45%, rgba(201,168,76,0.0) 90%);
    background-size: 200% 100%;
    animation: grad-shimmer 6s linear infinite;
}

/* ─────────────────────────────────────────────
   EXTRA DESIGN & INTERACTIVITY PASS
   Purely visual/behavioral polish layered on top
   of the existing look — no structural changes.
───────────────────────────────────────────── */

/* Custom gradient scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #c9a84c, #4a90d9);
    border-radius: 10px;
    border: 2px solid #0d0f14;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #e0bf6a, #6fb0f0); }

/* Cursor-reactive ambient glow that follows the mouse across the whole app */
#grad-cursor-glow {
    position: fixed;
    top: 0; left: 0;
    width: 480px; height: 480px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    background: radial-gradient(circle, rgba(201,168,76,0.10) 0%, rgba(74,144,217,0.05) 45%, rgba(0,0,0,0) 72%);
    transform: translate(-9999px, -9999px);
    transition: transform 0.06s linear, opacity 0.4s ease;
    mix-blend-mode: screen;
}

/* Headings — animated gold underline sweep on hover */
h1, h2, h3 { position: relative; }
h2::after {
    content: "";
    display: block;
    width: 0%;
    height: 2px;
    margin-top: 4px;
    background: linear-gradient(90deg, #c9a84c, transparent);
    transition: width 0.5s cubic-bezier(.2,.8,.2,1);
}
h2:hover::after { width: 38%; }

/* Buttons — magnetic 3D press + shine sweep */
.stButton > button {
    position: relative;
    overflow: hidden;
    transform-style: preserve-3d;
    transition: transform 0.18s cubic-bezier(.2,.8,.2,1), box-shadow 0.18s ease, background 0.2s ease !important;
}
.stButton > button::before {
    content: "";
    position: absolute;
    top: 0; left: -60%;
    width: 40%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.55), transparent);
    transform: skewX(-20deg);
    transition: left 0.55s ease;
    pointer-events: none;
}
.stButton > button:hover::before { left: 130%; }
.stButton > button:active {
    transform: perspective(400px) translateZ(-3px) scale(0.97) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}

/* Tabs — lift + glow on hover, 3D active indicator */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #181c26 !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid #252834 !important;
    border-bottom: none !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease, color 0.25s ease !important;
    transform-style: preserve-3d;
}
.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-3px) perspective(600px) rotateX(6deg);
    box-shadow: 0 8px 18px rgba(201,168,76,0.15);
}
.stTabs [aria-selected="true"] {
    box-shadow: 0 -2px 0 0 #c9a84c inset;
}

/* Expander — 3D chevron rotation + hover lift */
.streamlit-expanderHeader {
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
}
.streamlit-expanderHeader:hover {
    transform: perspective(700px) rotateX(1.5deg) translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    border-color: #c9a84c !important;
}

/* Text inputs / number inputs / sliders — focus glow ring with depth */
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    box-shadow: 0 0 0 3px rgba(201,168,76,0.22), 0 4px 14px rgba(0,0,0,0.35) !important;
    border-color: #c9a84c !important;
    transform: translateZ(2px);
}
.stSlider [role="slider"] {
    box-shadow: 0 0 0 4px rgba(201,168,76,0.18), 0 2px 8px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
.stSlider [role="slider"]:hover { transform: scale(1.15); }

/* Chat input — glowing focus bar */
[data-testid="stChatInput"] {
    transition: box-shadow 0.3s ease;
    border-radius: 10px;
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 2px rgba(201,168,76,0.35), 0 8px 26px rgba(201,168,76,0.12);
}

/* Chat message entrance — 3D swing-in */
@keyframes grad-msg-in {
    0%   { opacity: 0; transform: perspective(800px) rotateX(-12deg) translateY(14px); }
    100% { opacity: 1; transform: perspective(800px) rotateX(0deg) translateY(0); }
}
[data-testid="stChatMessage"] {
    animation: grad-msg-in 0.45s cubic-bezier(.2,.8,.2,1) both;
}

/* Landing-page feature cards — full 3D flip-lift on hover (targets inline style cards) */
div[style*="width:220px"] {
    transition: transform 0.45s cubic-bezier(.2,.8,.2,1), box-shadow 0.45s ease !important;
    transform-style: preserve-3d;
    cursor: default;
}
div[style*="width:220px"]:hover {
    transform: perspective(1000px) rotateY(8deg) rotateX(4deg) translateY(-8px) scale(1.03) !important;
    box-shadow: -12px 22px 40px rgba(0,0,0,0.5), 0 0 30px rgba(201,168,76,0.15) !important;
}

/* Metric widgets — subtle 3D pop */
.stMetric {
    transition: transform 0.3s cubic-bezier(.2,.8,.2,1), box-shadow 0.3s ease;
}
.stMetric:hover {
    transform: perspective(700px) rotateX(3deg) translateY(-3px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.4);
}

/* Progress bars — animated gold gradient fill */
.stProgress > div > div {
    background: linear-gradient(90deg, #c9a84c, #e0bf6a, #c9a84c) !important;
    background-size: 200% 100% !important;
    animation: grad-shimmer 3s linear infinite !important;
}

/* Checkbox — pop on check */
.stCheckbox label span {
    transition: transform 0.2s cubic-bezier(.34,1.56,.64,1);
}
.stCheckbox input:checked + span,
.stCheckbox [data-checked="true"] {
    transform: scale(1.12);
}

/* Radio pills — 3D depth on selected */
[data-testid="stRadio"] label {
    transition: transform 0.2s ease;
}
[data-testid="stRadio"] label:hover { transform: translateY(-1px); }

/* Divider — animated gold shimmer line instead of flat hr */
hr {
    background: linear-gradient(90deg, transparent, #c9a84c, transparent) !important;
    height: 1px !important;
    border: none !important;
    opacity: 0.5;
}

/* Score badge — parallax-reactive sheen overlay */
.score-badge {
    position: relative;
    overflow: hidden;
}
.score-badge::after {
    content: "";
    position: absolute;
    top: -50%; left: -60%;
    width: 40%; height: 200%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.5), transparent);
    transform: rotate(20deg);
    animation: grad-badge-sheen 3.2s ease-in-out infinite;
}
@keyframes grad-badge-sheen {
    0%   { left: -60%; }
    45%  { left: 130%; }
    100% { left: 130%; }
}

/* ─────────────────────────────────────────────
   HOLOGRAPHIC "POP-OUT" TREATMENT
   Makes key panels feel like they're projected
   above the page surface — scanlines, chromatic
   rim glow, and a floating drop-shadow that
   reads as depth toward the viewer.
───────────────────────────────────────────── */

@keyframes grad-holo-scan {
    0%   { background-position: 0 -100%; }
    100% { background-position: 0 200%; }
}
@keyframes grad-holo-flicker {
    0%, 100% { opacity: 1; }
    92%      { opacity: 1; }
    93%      { opacity: 0.82; }
    94%      { opacity: 1; }
    97%      { opacity: 0.9; }
}
@keyframes grad-holo-hover {
    0%, 100% { transform: perspective(1000px) rotateX(4deg) translateY(0) translateZ(18px); }
    50%      { transform: perspective(1000px) rotateX(4deg) translateY(-9px) translateZ(26px); }
}

/* Score badge becomes a projected holographic readout */
.score-badge {
    background: linear-gradient(135deg, #c9a84c, #e0bf6a) !important;
    box-shadow:
        0 0 0 1px rgba(201,168,76,0.5),
        0 -1px 0 rgba(74,144,217,0.6),
        0 1px 0 rgba(224,80,80,0.35),
        0 30px 55px -10px rgba(0,0,0,0.65),
        0 0 40px rgba(201,168,76,0.35) !important;
    animation: grad-badge-pop 0.7s cubic-bezier(.2,.8,.2,1) both,
               grad-holo-hover 5s ease-in-out 0.7s infinite,
               grad-holo-flicker 7s linear infinite !important;
}
.score-badge::before {
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(13,15,20,0.10) 0px,
        rgba(13,15,20,0.10) 1px,
        transparent 2px,
        transparent 4px
    );
    background-size: 100% 200%;
    animation: grad-holo-scan 2.6s linear infinite;
    pointer-events: none;
    mix-blend-mode: multiply;
}

/* Accent cards read as holographic panels floating above the surface */
.grad-card-accent {
    position: relative;
    transform: perspective(1000px) translateZ(0);
}
.grad-card-accent::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 12px;
    background: repeating-linear-gradient(
        to bottom,
        rgba(74,144,217,0.05) 0px,
        rgba(74,144,217,0.05) 1px,
        transparent 3px,
        transparent 6px
    );
    background-size: 100% 220%;
    animation: grad-holo-scan 3.4s linear infinite;
    pointer-events: none;
}
.grad-card-accent::after {
    content: "";
    position: absolute;
    left: 6%; right: 6%; bottom: -14px;
    height: 22px;
    border-radius: 50%;
    background: radial-gradient(ellipse at center, rgba(201,168,76,0.28) 0%, transparent 75%);
    filter: blur(3px);
    pointer-events: none;
    transition: opacity 0.4s ease;
}
.grad-card-accent:hover {
    transform: perspective(1000px) rotateX(3deg) rotateY(-3deg) translateY(-10px) translateZ(30px) !important;
    box-shadow:
        0 -1px 0 rgba(74,144,217,0.5),
        0 1px 0 rgba(224,80,80,0.30),
        0 40px 70px -12px rgba(0,0,0,0.7),
        0 0 45px rgba(201,168,76,0.22) !important;
    border-color: #c9a84c !important;
}

/* Regular cards get a subtler chromatic rim + lift-toward-viewer on hover */
.grad-card:hover {
    box-shadow:
        0 22px 45px rgba(0,0,0,0.45),
        -1px 0 0 rgba(224,80,80,0.25),
        1px 0 0 rgba(74,144,217,0.25),
        0 0 28px rgba(201,168,76,0.12) !important;
}

/* Chat bubbles get a faint holographic edge + scan sweep */
[data-testid="stChatMessage"] {
    position: relative;
}
[data-testid="stChatMessage"]::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 12px;
    background: repeating-linear-gradient(
        to bottom,
        rgba(201,168,76,0.03) 0px,
        rgba(201,168,76,0.03) 1px,
        transparent 3px,
        transparent 7px
    );
    background-size: 100% 240%;
    animation: grad-holo-scan 4s linear infinite;
    pointer-events: none;
}

/* Global holographic vertical sweep across the entire app — very faint, reads as a projector refresh */
body::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, rgba(74,144,217,0.35), rgba(201,168,76,0.35), transparent);
    z-index: 999999;
    pointer-events: none;
    animation: grad-holo-sweep 9s linear infinite;
    opacity: 0.55;
}
@keyframes grad-holo-sweep {
    0%   { top: -3px; }
    100% { top: 100vh; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3D SCENERY  —  WebGL starfield + drifting "GradPath" constellation
# Injected once into the page background (behind all content),
# fully decorative, doesn't alter any app structure or logic.
# ─────────────────────────────────────────────
_GRADPATH_3D_BG = r"""
<div id="gradpath-3d-holder" style="width:0;height:0;overflow:hidden"></div>
<div id="grad-cursor-glow"></div>
<script>
(function () {
    // ── Cursor-reactive ambient glow (lightweight, pure DOM/CSS) ──
    function initCursorGlow(doc, win) {
        var glow = doc.getElementById('grad-cursor-glow');
        if (!glow || glow.dataset.gradBound === '1') return;
        glow.dataset.gradBound = '1';
        win.addEventListener('mousemove', function (e) {
            glow.style.transform = 'translate(' + (e.clientX - 240) + 'px,' + (e.clientY - 240) + 'px)';
        });
        win.addEventListener('mouseleave', function () {
            glow.style.transform = 'translate(-9999px, -9999px)';
        });
    }

    function buildScene(doc, win) {
        if (doc.getElementById('gradpath-3d-bg')) return;

        var canvas = doc.createElement('canvas');
        canvas.id = 'gradpath-3d-bg';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.zIndex = '-1';
        canvas.style.pointerEvents = 'none';
        canvas.style.display = 'block';
        doc.body.appendChild(canvas);

        var THREE = win.THREE;
        var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(win.devicePixelRatio || 1, 2));
        renderer.setSize(win.innerWidth, win.innerHeight);

        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(58, win.innerWidth / win.innerHeight, 0.1, 1000);
        camera.position.z = 55;

        // ── Starfield (gold + soft blue, academic "night sky") ──
        var starCount = 900;
        var positions = new Float32Array(starCount * 3);
        var colors = new Float32Array(starCount * 3);
        for (var i = 0; i < starCount; i++) {
            positions[i * 3]     = (Math.random() - 0.5) * 320;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 320;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 320;
            if (Math.random() > 0.72) {
                colors[i * 3] = 0.79; colors[i * 3 + 1] = 0.66; colors[i * 3 + 2] = 0.30; // gold
            } else {
                colors[i * 3] = 0.55; colors[i * 3 + 1] = 0.58; colors[i * 3 + 2] = 0.70; // soft blue-grey
            }
        }
        var starGeo = new THREE.BufferGeometry();
        starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        starGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        var starMat = new THREE.PointsMaterial({ size: 0.55, vertexColors: true, transparent: true, opacity: 0.85 });
        var stars = new THREE.Points(starGeo, starMat);
        scene.add(stars);

        // ── Gold wireframe icosahedron — "diploma seal / globe of opportunity" ──
        var icoGeo = new THREE.IcosahedronGeometry(13, 1);
        var icoMat = new THREE.MeshBasicMaterial({ color: 0xc9a84c, wireframe: true, transparent: true, opacity: 0.22 });
        var ico = new THREE.Mesh(icoGeo, icoMat);
        ico.position.set(26, 9, -35);
        scene.add(ico);

        // ── Blue torus — orbiting "campus ring" ──
        var torusGeo = new THREE.TorusGeometry(7.5, 0.35, 16, 60);
        var torusMat = new THREE.MeshBasicMaterial({ color: 0x4a90d9, wireframe: true, transparent: true, opacity: 0.16 });
        var torus = new THREE.Mesh(torusGeo, torusMat);
        torus.position.set(-29, -11, -22);
        scene.add(torus);

        // ── "GradPath" — a winding line of glowing nodes from bottom-left to top-right ──
        var pathPoints = [];
        for (var p = 0; p <= 10; p++) {
            var t = p / 10;
            pathPoints.push(new THREE.Vector3(
                -60 + t * 120 + Math.sin(t * Math.PI * 1.5) * 10,
                -35 + t * 70 + Math.cos(t * Math.PI) * 6,
                -50 + t * 20
            ));
        }
        var pathCurve = new THREE.CatmullRomCurve3(pathPoints);
        var pathGeo = new THREE.TubeGeometry(pathCurve, 64, 0.18, 6, false);
        var pathMat = new THREE.MeshBasicMaterial({ color: 0xc9a84c, transparent: true, opacity: 0.20 });
        var pathMesh = new THREE.Mesh(pathGeo, pathMat);
        scene.add(pathMesh);

        var nodeGeo = new THREE.SphereGeometry(0.55, 12, 12);
        var nodeMat = new THREE.MeshBasicMaterial({ color: 0xe0bf6a, transparent: true, opacity: 0.55 });
        var nodes = [];
        pathPoints.forEach(function (pt) {
            var node = new THREE.Mesh(nodeGeo, nodeMat);
            node.position.copy(pt);
            scene.add(node);
            nodes.push(node);
        });

        // ── Constellation links — faint lines connecting nearby stars for a "network" feel ──
        var linkPositions = [];
        var linkSampleCount = 140;
        for (var li = 0; li < linkSampleCount; li++) {
            var a = Math.floor(Math.random() * starCount);
            var b = Math.floor(Math.random() * starCount);
            var ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
            var bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
            var dist = Math.sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz));
            if (dist < 26) {
                linkPositions.push(ax, ay, az, bx, by, bz);
            }
        }
        var linkGeo = new THREE.BufferGeometry();
        linkGeo.setAttribute('position', new THREE.Float32BufferAttribute(linkPositions, 3));
        var linkMat = new THREE.LineBasicMaterial({ color: 0x4a90d9, transparent: true, opacity: 0.09 });
        var links = new THREE.LineSegments(linkGeo, linkMat);
        scene.add(links);

        // ── Floating "graduation cap" — simple primitive composition (board + button + tassel) ──
        var capGroup = new THREE.Group();
        var boardGeo = new THREE.BoxGeometry(6, 0.35, 6);
        var boardMat = new THREE.MeshBasicMaterial({ color: 0x181c26, transparent: true, opacity: 0.55 });
        var board = new THREE.Mesh(boardGeo, boardMat);
        capGroup.add(board);

        var boardEdges = new THREE.LineSegments(
            new THREE.EdgesGeometry(boardGeo),
            new THREE.LineBasicMaterial({ color: 0xc9a84c, transparent: true, opacity: 0.6 })
        );
        capGroup.add(boardEdges);

        var domeGeo = new THREE.SphereGeometry(1.6, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2);
        var domeMat = new THREE.MeshBasicMaterial({ color: 0x181c26, wireframe: true, transparent: true, opacity: 0.4 });
        var dome = new THREE.Mesh(domeGeo, domeMat);
        dome.position.y = 0.18;
        capGroup.add(dome);

        var buttonGeo = new THREE.SphereGeometry(0.22, 8, 8);
        var buttonMat = new THREE.MeshBasicMaterial({ color: 0xc9a84c, transparent: true, opacity: 0.8 });
        var button = new THREE.Mesh(buttonGeo, buttonMat);
        button.position.y = 0.2;
        capGroup.add(button);

        var tasselCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0, 0.2, 0),
            new THREE.Vector3(2.4, -0.4, 0.4),
            new THREE.Vector3(2.9, -2.4, 0.6),
        ]);
        var tasselGeo = new THREE.TubeGeometry(tasselCurve, 20, 0.06, 6, false);
        var tasselMat = new THREE.MeshBasicMaterial({ color: 0xc9a84c, transparent: true, opacity: 0.7 });
        var tassel = new THREE.Mesh(tasselGeo, tasselMat);
        capGroup.add(tassel);

        capGroup.position.set(-14, 22, -30);
        capGroup.rotation.x = 0.3;
        scene.add(capGroup);

        // ── Additional floating gem shapes for depth/parallax layering ──
        var gems = [];
        var gemDefs = [
            { geo: new THREE.OctahedronGeometry(2.2, 0), color: 0xc9a84c, pos: [42, -22, -18] },
            { geo: new THREE.TetrahedronGeometry(2.6, 0), color: 0x4a90d9, pos: [-40, 24, -40] },
            { geo: new THREE.DodecahedronGeometry(1.9, 0), color: 0xe0bf6a, pos: [8, -30, -28] },
        ];
        gemDefs.forEach(function (def) {
            var mat = new THREE.MeshBasicMaterial({ color: def.color, wireframe: true, transparent: true, opacity: 0.24 });
            var mesh = new THREE.Mesh(def.geo, mat);
            mesh.position.set(def.pos[0], def.pos[1], def.pos[2]);
            scene.add(mesh);
            gems.push(mesh);
        });

        // ── HOLOGRAM PROJECTOR — a glowing base ring with a translucent light
        //    cone rising from it, plus concentric scan-rings drifting upward,
        //    reading like a classic sci-fi holographic projection. ──
        var holoGroup = new THREE.Group();
        holoGroup.position.set(20, -14, -12);

        var holoBaseGeo = new THREE.RingGeometry(2.6, 3.0, 48);
        var holoBaseMat = new THREE.MeshBasicMaterial({
            color: 0x4a90d9, transparent: true, opacity: 0.55,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending
        });
        var holoBase = new THREE.Mesh(holoBaseGeo, holoBaseMat);
        holoBase.rotation.x = -Math.PI / 2;
        holoGroup.add(holoBase);

        var holoConeGeo = new THREE.ConeGeometry(2.7, 11, 32, 1, true);
        var holoConeMat = new THREE.MeshBasicMaterial({
            color: 0xc9a84c, transparent: true, opacity: 0.055,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
        });
        var holoCone = new THREE.Mesh(holoConeGeo, holoConeMat);
        holoCone.position.y = 5.5;
        holoGroup.add(holoCone);

        // Concentric scan-rings that drift up through the beam like a hologram refresh
        var scanRings = [];
        for (var sr = 0; sr < 4; sr++) {
            var scanGeo = new THREE.RingGeometry(1.4, 1.55, 40);
            var scanMat = new THREE.MeshBasicMaterial({
                color: 0xe0bf6a, transparent: true, opacity: 0.5,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
            });
            var scanRing = new THREE.Mesh(scanGeo, scanMat);
            scanRing.rotation.x = -Math.PI / 2;
            scanRing.userData.offset = sr / 4;
            holoGroup.add(scanRing);
            scanRings.push(scanRing);
        }
        scene.add(holoGroup);

        // ── Orbiting HUD rings around the graduation cap — reads as a scanner halo ──
        var hudRings = [];
        var hudRingDefs = [
            { r: 3.6, color: 0xc9a84c, tilt: [1.1, 0, 0] },
            { r: 4.3, color: 0x4a90d9, tilt: [0, 0.9, 0.4] },
            { r: 5.0, color: 0xe0bf6a, tilt: [0.6, 0.6, 0] },
        ];
        hudRingDefs.forEach(function (def) {
            var g = new THREE.RingGeometry(def.r, def.r + 0.06, 64);
            var m = new THREE.MeshBasicMaterial({
                color: def.color, transparent: true, opacity: 0.28,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
            });
            var ring = new THREE.Mesh(g, m);
            ring.rotation.set(def.tilt[0], def.tilt[1], def.tilt[2]);
            capGroup.add(ring);
            hudRings.push(ring);
        });

        // ── Floating holographic "data panels" — thin wireframe grid planes that
        //    hover in space like projected HUD readouts, each with a soft glow face ──
        var holoPanels = [];
        var panelDefs = [
            { pos: [-34, 6, -18], size: [9, 6], color: 0x4a90d9 },
            { pos: [34, 18, -30], size: [7, 5], color: 0xc9a84c },
        ];
        panelDefs.forEach(function (def) {
            var group = new THREE.Group();

            var faceGeo = new THREE.PlaneGeometry(def.size[0], def.size[1]);
            var faceMat = new THREE.MeshBasicMaterial({
                color: def.color, transparent: true, opacity: 0.045,
                side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false
            });
            var face = new THREE.Mesh(faceGeo, faceMat);
            group.add(face);

            var gridGeo = new THREE.PlaneGeometry(def.size[0], def.size[1], 6, 4);
            var gridWire = new THREE.LineSegments(
                new THREE.WireframeGeometry(gridGeo),
                new THREE.LineBasicMaterial({ color: def.color, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending })
            );
            group.add(gridWire);

            var borderGeo = new THREE.EdgesGeometry(faceGeo);
            var border = new THREE.LineSegments(
                borderGeo,
                new THREE.LineBasicMaterial({ color: def.color, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending })
            );
            group.add(border);

            group.position.set(def.pos[0], def.pos[1], def.pos[2]);
            scene.add(group);
            holoPanels.push(group);
        });

        // ── Click-triggered gold particle burst ──
        var bursts = [];
        function spawnBurst(worldPos) {
            var count = 40;
            var geo = new THREE.BufferGeometry();
            var pos = new Float32Array(count * 3);
            var vel = [];
            for (var i = 0; i < count; i++) {
                pos[i*3] = worldPos.x; pos[i*3+1] = worldPos.y; pos[i*3+2] = worldPos.z;
                vel.push(new THREE.Vector3(
                    (Math.random() - 0.5) * 0.6,
                    (Math.random() - 0.5) * 0.6,
                    (Math.random() - 0.5) * 0.6
                ));
            }
            geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            var mat = new THREE.PointsMaterial({ color: 0xe0bf6a, size: 0.5, transparent: true, opacity: 1 });
            var pts = new THREE.Points(geo, mat);
            scene.add(pts);
            bursts.push({ points: pts, vel: vel, life: 0 });
        }
        win.addEventListener('click', function (e) {
            var vec = new THREE.Vector3(
                (e.clientX / win.innerWidth) * 2 - 1,
                -(e.clientY / win.innerHeight) * 2 + 1,
                0.5
            );
            vec.unproject(camera);
            var dir = vec.sub(camera.position).normalize();
            var dist = 40;
            var pos = camera.position.clone().add(dir.multiplyScalar(dist));
            spawnBurst(pos);
        });

        // ── Mouse parallax ──
        var mouseX = 0, mouseY = 0;
        win.addEventListener('mousemove', function (e) {
            mouseX = (e.clientX / win.innerWidth - 0.5);
            mouseY = (e.clientY / win.innerHeight - 0.5);
        });

        // ── Scroll-linked camera drift for a subtle parallax-on-scroll feel ──
        var scrollT = 0;
        function bindScroll() {
            var scroller = doc.querySelector('section.main') || doc.scrollingElement || doc.body;
            function onScroll() {
                var max = (scroller.scrollHeight - scroller.clientHeight) || 1;
                scrollT = Math.min(1, Math.max(0, scroller.scrollTop / max));
            }
            scroller.addEventListener('scroll', onScroll, { passive: true });
        }
        try { bindScroll(); } catch (e) {}

        var clock = new THREE.Clock();
        function animate() {
            requestAnimationFrame(animate);
            var t = clock.getElapsedTime();

            stars.rotation.y += 0.0006;
            stars.rotation.x += 0.0002;
            links.rotation.y += 0.0006;
            links.rotation.x += 0.0002;

            ico.rotation.x += 0.0025;
            ico.rotation.y += 0.0035;

            torus.rotation.x += 0.0020;
            torus.rotation.y -= 0.0018;

            capGroup.rotation.y += 0.0028;
            capGroup.position.y = 22 + Math.sin(t * 0.6) * 1.6;

            gems.forEach(function (g, gi) {
                g.rotation.x += 0.002 + gi * 0.0006;
                g.rotation.y += 0.0025 + gi * 0.0004;
                g.position.y += Math.sin(t * 0.5 + gi) * 0.01;
            });

            // ── Hologram projector: base pulses, cone shimmers, scan-rings rise through the beam ──
            holoBase.material.opacity = 0.4 + 0.25 * Math.sin(t * 2.2);
            holoBase.scale.setScalar(1 + 0.04 * Math.sin(t * 2.2));
            holoCone.material.opacity = 0.04 + 0.03 * Math.sin(t * 1.4);
            scanRings.forEach(function (ring) {
                var cycle = ((t * 0.18 + ring.userData.offset) % 1);
                ring.position.y = cycle * 11;
                ring.scale.setScalar(0.6 + cycle * 1.1);
                ring.material.opacity = 0.55 * (1 - cycle);
            });
            holoGroup.rotation.y += 0.003;

            // ── HUD scanner rings orbiting the graduation cap, each on its own axis ──
            hudRings.forEach(function (ring, ri) {
                ring.rotation.z += 0.004 + ri * 0.0015;
                ring.material.opacity = 0.18 + 0.14 * Math.sin(t * 1.6 + ri * 1.3);
            });

            // ── Floating holographic data panels — gentle bob + slow yaw, like projected HUDs ──
            holoPanels.forEach(function (panel, pi) {
                panel.rotation.y = Math.sin(t * 0.25 + pi) * 0.35;
                panel.position.y += Math.sin(t * 0.4 + pi * 2) * 0.006;
            });

            nodes.forEach(function (node, idx) {
                node.scale.setScalar(1 + 0.25 * Math.sin(t * 1.5 + idx * 0.6));
            });

            // Animate + retire particle bursts
            for (var bi = bursts.length - 1; bi >= 0; bi--) {
                var b = bursts[bi];
                var arr = b.points.geometry.attributes.position.array;
                for (var pi = 0; pi < b.vel.length; pi++) {
                    arr[pi*3]   += b.vel[pi].x;
                    arr[pi*3+1] += b.vel[pi].y - 0.01;
                    arr[pi*3+2] += b.vel[pi].z;
                }
                b.points.geometry.attributes.position.needsUpdate = true;
                b.life += 0.02;
                b.points.material.opacity = Math.max(0, 1 - b.life);
                if (b.life >= 1) {
                    scene.remove(b.points);
                    bursts.splice(bi, 1);
                }
            }

            camera.position.x += (mouseX * 10 - camera.position.x) * 0.02;
            camera.position.y += (-mouseY * 10 - 6 * scrollT - camera.position.y) * 0.02;
            camera.fov = 58 + scrollT * 4;
            camera.updateProjectionMatrix();
            camera.lookAt(scene.position);

            renderer.render(scene, camera);
        }
        animate();

        win.addEventListener('resize', function () {
            renderer.setSize(win.innerWidth, win.innerHeight);
            camera.aspect = win.innerWidth / win.innerHeight;
            camera.updateProjectionMatrix();
        });
    }

    function init() {
        try {
            var targetWin = (window.parent && window.parent !== window) ? window.parent : window;
            var targetDoc = targetWin.document;

            initCursorGlow(targetDoc, targetWin);

            if (targetWin.THREE) {
                buildScene(targetDoc, targetWin);
                return;
            }
            var script = targetDoc.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.128.0/three.min.js';
            script.onload = function () { buildScene(targetDoc, targetWin); };
            targetDoc.head.appendChild(script);
        } catch (err) {
            // Cross-origin fallback: render confined to this component's iframe
            initCursorGlow(document, window);
            if (window.THREE) {
                buildScene(document, window);
                return;
            }
            var s = document.createElement('script');
            s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.128.0/three.min.js';
            s.onload = function () { buildScene(document, window); };
            document.head.appendChild(s);
        }
    }
    init();
})();
</script>
"""
components.html(_GRADPATH_3D_BG, height=0, width=0)

# ─────────────────────────────────────────────
# FILE STORAGE
# ─────────────────────────────────────────────
DATA_DIR = "data"
PROFILE_FILE = os.path.join(DATA_DIR, "profiles.json")
CHAT_FILE    = os.path.join(DATA_DIR, "chats.json")
CHECKLIST_FILE = os.path.join(DATA_DIR, "checklists.json")
os.makedirs(DATA_DIR, exist_ok=True)

def load_json(file):
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

profiles  = load_json(PROFILE_FILE)
chats     = load_json(CHAT_FILE)
checklists = load_json(CHECKLIST_FILE)

# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register(username: str, password: str) -> tuple[bool, str]:
    if not username.strip():
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username in profiles:
        return False, "Username already exists. Please log in."
    profiles[username] = {
        "password": hash_pw(password),
        "created":  str(datetime.datetime.now()),
        "academic": {}
    }
    save_json(PROFILE_FILE, profiles)
    return True, "Account created!"

def login(username: str, password: str) -> tuple[bool, str]:
    if username not in profiles:
        return False, "Username not found."
    if profiles[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, "ok"

def logout():
    for key in ["user", "current_chat", "chat_id", "last_prediction"]:
        st.session_state.pop(key, None)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "user":            None,
    "current_chat":    [],
    "chat_id":         None,
    "last_prediction": None,
    "auth_tab":        "Login",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# PREDICTION ENGINE  —  real ML model
# ─────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

def _find_dataset():
    for name in ["dataset.csv", "dataset", "Admission_Predict_Ver1.1.csv", "Admission_Predict.csv"]:
        p = os.path.join("model", name)
        if os.path.exists(p):
            print(f"GRADPATH — Found dataset: {os.path.abspath(p)}")
            return p
    print(f"GRADPATH — WARNING: No dataset found in model/ folder!")
    return os.path.join("model", "dataset.csv")

DATASET_PATH = _find_dataset()
FEATURE_COLS = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

# ── Engineered feature set for the upgraded ensemble model ──
# Built on top of the same 7 raw inputs — adds interaction terms and a
# composite "academic index" so the model can pick up on combined-factor
# signal (e.g. a high GRE paired with a high CGPA is stronger evidence
# than either alone).
ENGINEERED_FEATURE_NAMES = FEATURE_COLS + [
    "GRE_CGPA_interaction", "TOEFL_CGPA_interaction", "SOPLOR_avg", "Academic_Index"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive extra signal features on top of the 7 raw admission factors.

    Used identically at training time and at prediction time so the
    feature space the model sees always matches.
    """
    out = df.copy()
    gre_n      = (out["GRE Score"]   - 260) / 80.0
    toefl_n    =  out["TOEFL Score"] / 120.0
    cgpa_n     =  out["CGPA"]        / 10.0
    soplor_avg = (out["SOP"] + out["LOR"]) / 2.0

    out["GRE_CGPA_interaction"]   = gre_n * cgpa_n
    out["TOEFL_CGPA_interaction"] = toefl_n * cgpa_n
    out["SOPLOR_avg"]             = soplor_avg
    out["Academic_Index"] = (
        gre_n * 0.30 + toefl_n * 0.15 + cgpa_n * 0.30 +
        (soplor_avg / 5.0) * 0.15 + out["Research"].astype(float) * 0.10
    )
    return out[ENGINEERED_FEATURE_NAMES]

@st.cache_resource(show_spinner="Training model on your dataset…")
def load_or_train_model():
    os.makedirs("model", exist_ok=True)

    print("=" * 50)
    print("GRADPATH — MODEL LOADER")
    print(f"  Model exists?  : {os.path.exists(MODEL_PATH)}")
    print(f"  Scaler exists? : {os.path.exists(SCALER_PATH)}")
    print("=" * 50)

    expected_features = len(ENGINEERED_FEATURE_NAMES)

    # ── Already trained — load from disk if it matches the current feature set ──
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH,  "rb") as f: cached_model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: cached_scaler = pickle.load(f)

            if getattr(cached_scaler, "n_features_in_", None) == expected_features:
                print("✅ Pre-trained model found — loading from disk (skipping training)")
                print("=" * 50)
                return cached_model, cached_scaler
            else:
                print("⚠ Cached model uses an outdated feature set — retraining with the upgraded pipeline...")
        except Exception as e:
            print(f"⚠ Could not load cached model ({e}) — retraining...")

    # ── Load all datasets from model/ folder ──
    dfs = []
    for fname in os.listdir("model"):
        if fname.endswith(".csv"):
            path = os.path.join("model", fname)
            try:
                tmp = pd.read_csv(path)
                tmp.columns = tmp.columns.str.strip()
                if "Chance of Admit" in tmp.columns:
                    dfs.append(tmp)
                    print(f"📂 Loaded: {fname} ({len(tmp)} rows)")
            except Exception as e:
                print(f"⚠ Could not read {fname}: {e}")

    if not dfs:
        print("❌ ERROR: No valid dataset CSV found in model/ folder!")
        print(f"   Place dataset.csv in: {os.path.abspath('model')}")
        print("=" * 50)
        return None, None

    # ── Combine all datasets ──
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df.columns = df.columns.str.strip()
    print(f"✅ Combined total: {len(df)} rows after deduplication")

    target = "Chance of Admit"
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ ERROR: Missing columns: {missing}")
        print(f"   Columns found: {list(df.columns)}")
        print("=" * 50)
        return None, None

    X = engineer_features(df[FEATURE_COLS].astype(float))
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Split — {len(X_train)} training rows, {len(X_test)} test rows")
    print(f"✅ Feature set ({expected_features} features): {ENGINEERED_FEATURE_NAMES}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🧠 Training advanced stacked ensemble (Gradient Boosting + Random Forest + Extra Trees, "
          "blended by a Ridge meta-learner)...")

    base_estimators = [
        ("gbr", GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=3,
            subsample=0.85, random_state=42
        )),
        ("rf", RandomForestRegressor(
            n_estimators=400, max_depth=9, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
        ("etr", ExtraTreesRegressor(
            n_estimators=400, max_depth=11, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
    ]

    model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        passthrough=True,
        cv=5,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    pct_mae = round(mae * 100, 2)

    try:
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="neg_mean_absolute_error")
        cv_pct_mae = round(-cv_scores.mean() * 100, 2)
    except Exception as e:
        cv_pct_mae = None
        print(f"⚠ Cross-validation diagnostic skipped: {e}")

    print(f"✅ Training complete!")
    print(f"   Hold-out MAE: {round(mae, 4)} = ~{pct_mae}% average error (lower is better)")
    if cv_pct_mae is not None:
        print(f"   5-fold CV MAE: ~{cv_pct_mae}% average error")

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"✅ Model saved → {os.path.abspath(MODEL_PATH)}")
    print("=" * 50)

    return model, scaler


def predict_admission(gre: float, cgpa: float, toefl: float,
                      sop_lor: float, research: int, uni_rating: int) -> dict:

    # Always compute factor scores for bar chart and tips
    gre_score   = (gre   - 260) / 80
    cgpa_score  = cgpa / 10
    toefl_score = toefl / 120
    soplo_score = (sop_lor - 1) / 4
    res_score   = float(research)
    uni_score   = (uni_rating - 1) / 4

    model, scaler = load_or_train_model()

    if model is None:
        # ── Fallback math formula ──
        weighted = (gre_score*0.25 + cgpa_score*0.25 + toefl_score*0.15 +
                    soplo_score*0.15 + res_score*0.10 + uni_score*0.10)
        x    = (weighted - 0.6) * 10
        prob = max(0.02, min(0.98, 1/(1+math.exp(-x)) + random.uniform(-0.02, 0.02)))
    else:
        # Feature order MUST match FEATURE_COLS exactly:
        # ["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research"]
        # then passed through engineer_features() so the model sees the same
        # interaction / composite-index columns it was trained on.
        raw_df = pd.DataFrame([{
            "GRE Score": gre, "TOEFL Score": toefl, "University Rating": uni_rating,
            "SOP": sop_lor, "LOR": sop_lor, "CGPA": cgpa, "Research": research,
        }])
        features   = engineer_features(raw_df).to_numpy(dtype=float)
        features_s = scaler.transform(features)
        raw  = float(model.predict(features_s)[0])
        prob = max(0.02, min(0.98, raw))

    pct = round(prob * 100, 1)

    # Band label
    if pct >= 75:
        band, colour = "High", "#5cb85c"
    elif pct >= 50:
        band, colour = "Moderate", "#c9a84c"
    elif pct >= 30:
        band, colour = "Low–Moderate", "#e8963a"
    else:
        band, colour = "Low", "#d9534f"

    # Per-factor improvement tips
    tips = []
    if gre_score   < 0.6: tips.append("GRE score — aim for 310+")
    if cgpa_score  < 0.7: tips.append("CGPA — strengthen your academic record")
    if toefl_score < 0.7: tips.append("TOEFL — target 100+ for top programs")
    if soplo_score < 0.5: tips.append("SOP / LOR quality — invest time here")
    if res_score  == 0:   tips.append("Research experience — even one project helps")

    return {
        "probability": pct,
        "band":   band,
        "colour": colour,
        "tips":   tips,
        "factors": {
            "GRE":        round(gre_score   * 100, 1),
            "CGPA":       round(cgpa_score  * 100, 1),
            "TOEFL":      round(toefl_score * 100, 1),
            "SOP / LOR":  round(soplo_score * 100, 1),
            "Research":   round(res_score   * 100, 1),
            "Uni Rating": round(uni_score   * 100, 1),
        }
    }

# ─────────────────────────────────────────────
# CHAT MANAGEMENT
# ─────────────────────────────────────────────
def save_chat():
    user = st.session_state.user
    if not user: return
    cid = st.session_state.chat_id or str(datetime.datetime.now())
    chats.setdefault(user, {})[cid] = st.session_state.current_chat
    save_json(CHAT_FILE, chats)
    st.session_state.chat_id = cid

def load_chat(cid):
    user = st.session_state.user
    st.session_state.current_chat = chats[user][cid]
    st.session_state.chat_id = cid

def new_chat():
    st.session_state.current_chat = []
    st.session_state.chat_id = None

# ─────────────────────────────────────────────
# AI RESPONSE GENERATOR
# ─────────────────────────────────────────────

# ── University database keyed by profile tier ──
UNIVERSITIES = {
    "high": {                          # pct >= 75
        "cs": {
            "reach":  ["MIT (EECS)", "Stanford (CS)", "Carnegie Mellon (MSCS)", "UC Berkeley (EECS)"],
            "match":  ["UCLA (CS)", "UC San Diego (CS)", "Georgia Tech (CS)", "University of Michigan"],
            "safe":   ["Purdue (CS)", "UT Dallas (CS)", "Northeastern (Khoury)", "ASU (CS)"],
        },
        "ds": {
            "reach":  ["Stanford (Stats/DS)", "CMU (MSML)", "Columbia (DS)", "NYU (DS)"],
            "match":  ["UC San Diego (DS)", "Georgia Tech (Analytics)", "Purdue (DS)", "UMass Amherst"],
            "safe":   ["Northeastern (DS)", "ASU (DS)", "Indiana University", "UT Arlington"],
        },
        "ee": {
            "reach":  ["MIT (EECS)", "Stanford (EE)", "Caltech (EE)", "UC Berkeley (EECS)"],
            "match":  ["Georgia Tech (EE)", "Purdue (EE)", "University of Michigan (EE)", "UCLA (EE)"],
            "safe":   ["UT Dallas (EE)", "ASU (EE)", "Northeastern (EE)", "SUNY Stony Brook"],
        },
        "general": {
            "reach":  ["MIT", "Stanford", "Carnegie Mellon", "UC Berkeley"],
            "match":  ["UCLA", "UC San Diego", "Georgia Tech", "University of Michigan"],
            "safe":   ["Purdue", "Northeastern", "ASU", "UT Dallas"],
        },
    },
    "moderate": {                      # 50 <= pct < 75
        "cs": {
            "reach":  ["Georgia Tech (CS)", "Purdue (CS)", "UT Austin (CS)", "University of Michigan"],
            "match":  ["ASU (CS)", "Northeastern (CS)", "Texas A&M (CS)", "Indiana University"],
            "safe":   ["UT Dallas (CS)", "SUNY Buffalo (CS)", "Wayne State", "UMass Lowell"],
        },
        "ds": {
            "reach":  ["Georgia Tech (Analytics)", "Purdue (DS)", "UT Austin (Stats)", "Penn State"],
            "match":  ["ASU (DS)", "Northeastern (DS)", "Indiana University (DS)", "DePaul University"],
            "safe":   ["UT Arlington (DS)", "SUNY Buffalo", "Pace University", "Wilmington University"],
        },
        "ee": {
            "reach":  ["Georgia Tech (EE)", "Purdue (EE)", "UT Austin (EE)", "Penn State (EE)"],
            "match":  ["ASU (EE)", "Texas A&M (EE)", "University of Florida (EE)", "NC State (EE)"],
            "safe":   ["UT Arlington (EE)", "SUNY Buffalo (EE)", "University of Dayton", "Wichita State"],
        },
        "general": {
            "reach":  ["Georgia Tech", "Purdue", "UT Austin", "Penn State"],
            "match":  ["ASU", "Northeastern", "Texas A&M", "Indiana University"],
            "safe":   ["UT Dallas", "SUNY Buffalo", "Wayne State", "UMass Lowell"],
        },
    },
    "low": {                           # pct < 50
        "cs": {
            "reach":  ["ASU (CS)", "Northeastern (CS)", "Stevens (CS)", "DePaul (CS)"],
            "match":  ["SUNY Buffalo (CS)", "UT Dallas (CS)", "UMass Lowell (CS)", "Pace University"],
            "safe":   ["Wayne State (CS)", "Wichita State", "South Dakota State", "Texas A&M Commerce"],
        },
        "ds": {
            "reach":  ["ASU (DS)", "Northeastern (DS)", "DePaul (DS)", "Pace University (DS)"],
            "match":  ["SUNY Buffalo", "UT Arlington (DS)", "Wilmington University", "Bellevue University"],
            "safe":   ["Regis University", "Dakota State University", "American University (DS)", "Harrisburg University"],
        },
        "ee": {
            "reach":  ["ASU (EE)", "Stevens (EE)", "University of Dayton (EE)", "Wichita State (EE)"],
            "match":  ["SUNY Buffalo (EE)", "UT Arlington (EE)", "South Dakota State", "NC A&T (EE)"],
            "safe":   ["Morgan State", "Tennessee State", "Prairie View A&M", "Alabama A&M"],
        },
        "general": {
            "reach":  ["ASU", "Northeastern", "Stevens", "DePaul"],
            "match":  ["SUNY Buffalo", "UTD", "UMass Lowell", "Pace University"],
            "safe":   ["Wayne State", "Wichita State", "South Dakota State", "Harrisburg University"],
        },
    },
}

MAJORS = {
    "cs":   ["Computer Science", "Software Engineering", "Artificial Intelligence", "Cybersecurity", "Human-Computer Interaction"],
    "ds":   ["Data Science", "Machine Learning", "Business Analytics", "Biostatistics", "Computational Social Science"],
    "ee":   ["Electrical Engineering", "Computer Engineering", "Robotics", "Signal Processing", "Power Systems"],
    "bio":  ["Bioinformatics", "Computational Biology", "Biomedical Engineering", "Neuroscience", "Genomics"],
    "fin":  ["Financial Engineering", "Quantitative Finance", "Financial Mathematics", "Risk Management", "FinTech"],
    "mgmt": ["Management of Technology", "Engineering Management", "MBA (Tech Focus)", "Supply Chain", "Operations Research"],
}

def _get_tier(pct):
    if pct >= 75:  return "high"
    if pct >= 50:  return "moderate"
    return "low"

def _detect_field(msg):
    if any(w in msg for w in ["computer science", "cs", "software", "coding", "programming", "ai", "ml", "cyber"]):
        return "cs"
    if any(w in msg for w in ["data science", "data", "analytics", "machine learning", "statistics", "stat"]):
        return "ds"
    if any(w in msg for w in ["electrical", "ee", "electronics", "robotics", "circuit", "power"]):
        return "ee"
    if any(w in msg for w in ["biology", "bio", "biomedical", "genomics", "neuro"]):
        return "bio"
    if any(w in msg for w in ["finance", "financial", "quant", "fintech", "banking"]):
        return "fin"
    if any(w in msg for w in ["management", "mba", "business", "operations", "supply chain"]):
        return "mgmt"
    return "general"

def _university_response(pct, field):
    tier  = _get_tier(pct)
    unis  = UNIVERSITIES[tier]
    group = unis.get(field, unis["general"])
    reach = ", ".join(group["reach"])
    match = ", ".join(group["match"])
    safe  = ", ".join(group["safe"])
    field_label = {"cs":"Computer Science","ds":"Data Science","ee":"Electrical Engineering",
                   "bio":"Biology/Biomedical","fin":"Finance/Quant","mgmt":"Management","general":"your field"}.get(field,"your field")
    return (
        f"Based on your **{pct}%** predicted chance, here is a balanced list for **{field_label}**:\n\n"
        f"🎯 **Reach Schools** *(apply to 3–4)*\n  • {reach.replace(', ', chr(10)+'  • ')}\n\n"
        f"✅ **Match Schools** *(apply to 4–5)*\n  • {match.replace(', ', chr(10)+'  • ')}\n\n"
        f"🛡 **Safe Schools** *(apply to 2–3)*\n  • {safe.replace(', ', chr(10)+'  • ')}\n\n"
        f"💡 Aim for **10–12 applications total** spread across all three tiers. "
        f"Would you like tips on any specific school, or help with a different field?"
    )

def _major_response(msg, academic):
    cgpa = academic.get("cgpa", 7.0)
    gre  = academic.get("gre", 300)
    res  = academic.get("research", 0)

    suggestions = []

    # Quantitative strength → CS / DS / EE
    if gre >= 315 and cgpa >= 8.0:
        suggestions += [("Computer Science / AI", "Your strong GRE quant score and CGPA make you a great fit for top CS and AI programs.")]
        suggestions += [("Data Science / ML", "High analytical scores align well with Data Science and Machine Learning programs.")]
    elif gre >= 310:
        suggestions += [("Data Science", "Your GRE score suits quantitative programs like Data Science or Analytics.")]
        suggestions += [("Electrical Engineering", "Strong quant performance translates well to EE and Computer Engineering programs.")]

    # Research experience → research-heavy fields
    if res == 1:
        suggestions += [("Bioinformatics / Computational Biology", "Research experience is highly valued — Bioinformatics and Comp Bio programs reward it strongly.")]
        suggestions += [("Human-Computer Interaction", "Research background pairs well with HCI, a growing interdisciplinary MS field.")]

    # Lower quant / broader profile → management / finance
    if cgpa >= 7.0 and gre < 310:
        suggestions += [("Engineering Management / MBA", "A solid CGPA with a broader profile fits Management of Technology or MBA (Tech Focus) well.")]
        suggestions += [("Financial Engineering", "FinTech and Financial Engineering programs value analytical skills even with a moderate GRE.")]

    if not suggestions:
        suggestions = [
            ("Computer Science", "A versatile, in-demand degree with strong job market outcomes."),
            ("Data Science", "One of the fastest growing fields — suits quantitative and analytical thinkers."),
            ("Electrical Engineering", "Broad applicability across hardware, robotics, and embedded systems."),
        ]

    lines = "\n\n".join(f"**{name}**\n  {reason}" for name, reason in suggestions[:4])
    return (
        f"Based on your academic profile, here are some majors that could be a strong fit for you:\n\n"
        f"{lines}\n\n"
        "Would you like a university list tailored to any of these fields? Just ask!"
    )

def ai_response(user_msg: str, prediction: dict | None, academic: dict) -> str:
    msg  = user_msg.lower()
    pred = prediction

    if pred:
        pct      = pred["probability"]
        band     = pred["band"]
        tips     = pred["tips"]
        tips_str = "\n".join(f"  • {t}" for t in tips) if tips else "  • Your profile looks competitive across all factors!"
        field    = _detect_field(msg)

        # ── Major suggestions ──
        if any(w in msg for w in ["major", "field", "study", "speciali", "degree", "subject", "what should i"]):
            return _major_response(msg, academic)

        # ── University / school list ──
        if any(w in msg for w in ["university", "universities", "school", "college", "program", "apply", "list", "where", "which"]):
            return _university_response(pct, field)

        # ── General improvement tips ──
        if any(w in msg for w in ["improve", "boost", "increase", "better", "chance", "tip", "advice", "suggest", "how can i"]):
            uni_hint = _university_response(pct, field)
            return (
                f"Here are your highest-leverage improvements for a **{pct}% ({band})** profile:\n\n"
                f"{tips_str}\n\n"
                f"---\n\n{uni_hint}"
            )

        # ── GRE ──
        if any(w in msg for w in ["gre", "verbal", "quant", "score"]):
            g = academic.get("gre", 300)
            if g < 305:
                return (
                    f"Your GRE of **{g}** is below average for most MS programs (median is ~310–315).\n\n"
                    "**Action plan:**\n"
                    "  • Focus on **Quantitative Reasoning** — most STEM programs weight this heavily\n"
                    "  • Use Khan Academy + Manhattan Prep for quant fundamentals\n"
                    "  • Aim for 315+ quant, 155+ verbal\n"
                    "  • A 10-point improvement can shift your predicted chance by **5–10%**"
                )
            elif g < 320:
                return (
                    f"Your GRE of **{g}** is competitive for most programs.\n\n"
                    "To push into top-tier reach schools (MIT, Stanford, CMU), aim for **320+**. "
                    "One more attempt focusing on Quant could open significantly more doors."
                )
            return f"Your GRE of **{g}** is excellent — in the top tier. Focus your energy on SOP quality and research now."

        # ── CGPA ──
        if any(w in msg for w in ["cgpa", "gpa", "grade", "academic"]):
            c = academic.get("cgpa", 7.0)
            if c < 7.0:
                return (
                    f"A CGPA of **{c}/10** is below average for competitive programs (preferred: 8.0+).\n\n"
                    "**How to compensate:**\n"
                    "  • Highlight an **upward grade trend** in your SOP\n"
                    "  • Strong GRE quant (315+) can partially offset a lower CGPA\n"
                    "  • Research experience or publications carry significant weight\n"
                    "  • Consider applying to programs that emphasize work experience over grades"
                )
            elif c < 8.0:
                return (
                    f"Your CGPA of **{c}/10** is decent but below the sweet spot for top programs (8.0+).\n\n"
                    "Highlight your strongest relevant courses and any upward trend in grades in your SOP. "
                    "A strong GRE and research experience can more than compensate."
                )
            return f"Your CGPA of **{c}/10** is strong — well above average. Make sure your SOP reflects your top coursework."

        # ── SOP / LOR ──
        if any(w in msg for w in ["sop", "statement", "lor", "recommendation", "letter", "essay"]):
            return (
                "**SOP and LOR** are often the deciding factor for borderline applicants — here is how to make them count:\n\n"
                "**Statement of Purpose (SOP):**\n"
                "  • Open with a specific research problem or moment that sparked your interest\n"
                "  • Name specific faculty at each school you want to work with — shows genuine interest\n"
                "  • Quantify every achievement ('led a team of 4', 'improved accuracy by 12%')\n"
                "  • Keep it to 1–1.5 pages — concise beats comprehensive\n"
                "  • Tailor each SOP individually — generic SOPs are easy to spot\n\n"
                "**Letters of Recommendation (LOR):**\n"
                "  • Choose professors who have seen your work directly, not just your exam scores\n"
                "  • Brief your recommenders — give them your CV, SOP draft, and key points to mention\n"
                "  • Academic letters > professional letters for research-heavy programs\n"
                "  • Submit requests at least **6–8 weeks** before deadlines"
            )

        # ── Research ──
        if any(w in msg for w in ["research", "publication", "paper", "journal", "project"]):
            r = academic.get("research", 0)
            if r == 0:
                return (
                    "You currently have **no research experience** logged — this is worth addressing.\n\n"
                    "**Quick ways to build research experience:**\n"
                    "  • Email professors at your current institution asking to assist on a project\n"
                    "  • Join a lab as a volunteer or part-time research assistant\n"
                    "  • Start a self-directed project on Kaggle or GitHub with a clear research question\n"
                    "  • Apply for summer research programs (REUs in the US, similar abroad)\n"
                    "  • Even a conference poster or workshop paper adds real weight to your application\n\n"
                    "Research experience can shift your predicted chance by **+8–15%** for top programs."
                )
            return (
                "Research experience is one of your strongest assets — here is how to leverage it:\n\n"
                "  • Describe your **specific contribution** in the SOP, not just the project title\n"
                "  • If you have a publication or preprint, mention it prominently\n"
                "  • Ask your research supervisor for a LOR — it carries more weight than a course instructor\n"
                "  • Connect your research to why you want to pursue graduate study at each specific school"
            )

        # ── TOEFL ──
        if any(w in msg for w in ["toefl", "english", "language", "ielts"]):
            t = academic.get("toefl", 90)
            if t < 90:
                return (
                    f"Your TOEFL of **{t}** is below the minimum cutoff for many programs (90+).\n\n"
                    "This could result in automatic rejection before your academic profile is even reviewed. "
                    "Retaking the TOEFL should be your **top priority** right now.\n\n"
                    "  • Focus on the **Writing** and **Speaking** sections for the fastest gains\n"
                    "  • Target 100+ for most programs, 105+ for top-tier schools"
                )
            elif t < 100:
                return (
                    f"Your TOEFL of **{t}** clears the minimum but top programs prefer 100–110+.\n\n"
                    "  • A score of 100+ removes TOEFL as a weakness in your application\n"
                    "  • Writing section improvements tend to yield the biggest score jumps\n"
                    "  • Consider one retake if you have time before application deadlines"
                )
            return f"Your TOEFL of **{t}** is strong — above the threshold for virtually all programs. No action needed here."

        # ── Scholarship / funding ──
        if any(w in msg for w in ["scholarship", "funding", "fellowship", "financial", "aid", "cost", "money"]):
            return (
                "Here are the main funding options for international graduate students:\n\n"
                "**Fellowships & Scholarships:**\n"
                "  • **Fulbright Foreign Student Program** — highly competitive, covers full tuition\n"
                "  • **NSF Graduate Research Fellowship** — for US citizens/residents\n"
                "  • University-specific merit scholarships — check each school's graduate aid page\n\n"
                "**Assistantships (most common route):**\n"
                "  • **Research Assistantship (RA)** — work with a professor, covers tuition + stipend\n"
                "  • **Teaching Assistantship (TA)** — teach undergrad sections, covers tuition + stipend\n"
                "  • Apply directly to faculty whose research matches yours — cold emailing works\n\n"
                "**Tip:** PhD programs almost always come with full funding. "
                "If cost is a major concern, consider whether a PhD path fits your goals."
            )

        # ── Timeline / deadlines ──
        if any(w in msg for w in ["deadline", "timeline", "when", "apply", "semester", "fall", "spring"]):
            return (
                "**Typical US Graduate Admissions Timeline:**\n\n"
                "  • **June–August** — Research programs and professors, shortlist 10–15 schools\n"
                "  • **August–September** — Request LORs from recommenders (give 6–8 weeks notice)\n"
                "  • **September–October** — Draft and refine SOPs, take GRE/TOEFL if needed\n"
                "  • **October–November** — Submit early decision applications\n"
                "  • **December 1–15** — Most Fall intake deadlines fall here\n"
                "  • **January–February** — Rolling decisions begin arriving\n"
                "  • **April 15** — Standard acceptance deadline (Council of Graduate Schools)\n\n"
                "**Spring intake** deadlines are typically July–August of the prior year — fewer programs offer this."
            )

    # ── No prediction yet ──
    if any(w in msg for w in ["hello", "hi", "hey", "start", "begin"]):
        return (
            "Hello! 👋 Welcome to **GradPath AI**.\n\n"
            "To get started, fill in your academic profile on the left and hit **Run Prediction**. "
            "Once you have your results, you can ask me anything — university lists, major suggestions, "
            "how to improve your GRE, SOP tips, funding options, and more."
        )

    if any(w in msg for w in ["major", "field", "study", "degree", "what should"]):
        return "Run a prediction first using your academic profile on the left, then ask me about majors — I'll tailor suggestions to your specific scores!"

    if any(w in msg for w in ["university", "school", "college", "where", "apply"]):
        return "Run a prediction first and I'll give you a full tailored university list with reach, match, and safe schools for your chosen field!"

    return (
        "I can help with:\n\n"
        "  • 🏫 **University lists** — reach, match, and safe schools by field\n"
        "  • 🎓 **Major suggestions** — based on your profile strengths\n"
        "  • 📈 **Score improvement** — GRE, TOEFL, CGPA strategies\n"
        "  • ✍️ **SOP / LOR tips** — how to write a compelling application\n"
        "  • 💰 **Funding & scholarships** — assistantships and fellowships\n"
        "  • 📅 **Application timeline** — when to do what\n\n"
        "Run a prediction first, then ask me anything!"
    )

# ── Load model on startup so terminal shows output immediately ──
_startup_model, _startup_scaler = load_or_train_model()

# ═══════════════════════════════════════════════════════════
# ── NEW FEATURE 1: UNIVERSITY COMPARISON DATABASE ──
# ═══════════════════════════════════════════════════════════
UNIVERSITY_PROFILES = {
    # ── Elite ──
    "MIT":                    {"gre": 330, "cgpa": 9.5, "toefl": 113, "acceptance_rate": 4,  "research_req": True,  "rank": 1},
    "Stanford":               {"gre": 328, "cgpa": 9.3, "toefl": 112, "acceptance_rate": 5,  "research_req": True,  "rank": 2},
    "Carnegie Mellon":        {"gre": 325, "cgpa": 9.0, "toefl": 110, "acceptance_rate": 10, "research_req": True,  "rank": 3},
    "UC Berkeley":            {"gre": 327, "cgpa": 9.2, "toefl": 110, "acceptance_rate": 7,  "research_req": True,  "rank": 4},
    "Caltech":                {"gre": 331, "cgpa": 9.6, "toefl": 114, "acceptance_rate": 3,  "research_req": True,  "rank": 1},
    "Princeton":              {"gre": 329, "cgpa": 9.4, "toefl": 111, "acceptance_rate": 5,  "research_req": True,  "rank": 2},
    "Harvard":                {"gre": 328, "cgpa": 9.3, "toefl": 111, "acceptance_rate": 6,  "research_req": True,  "rank": 3},
    "Johns Hopkins":          {"gre": 323, "cgpa": 9.0, "toefl": 109, "acceptance_rate": 11, "research_req": True,  "rank": 9},
    # ── Highly Selective ──
    "UCLA":                   {"gre": 319, "cgpa": 8.8, "toefl": 106, "acceptance_rate": 16, "research_req": False, "rank": 6},
    "Columbia":               {"gre": 322, "cgpa": 8.9, "toefl": 108, "acceptance_rate": 12, "research_req": False, "rank": 8},
    "University of Michigan": {"gre": 318, "cgpa": 8.6, "toefl": 104, "acceptance_rate": 20, "research_req": False, "rank": 7},
    "Georgia Tech":           {"gre": 320, "cgpa": 8.7, "toefl": 105, "acceptance_rate": 18, "research_req": False, "rank": 5},
    "Cornell":                {"gre": 321, "cgpa": 8.8, "toefl": 107, "acceptance_rate": 14, "research_req": False, "rank": 7},
    "UC San Diego":           {"gre": 316, "cgpa": 8.4, "toefl": 103, "acceptance_rate": 24, "research_req": False, "rank": 11},
    "NYU":                    {"gre": 315, "cgpa": 8.5, "toefl": 105, "acceptance_rate": 22, "research_req": False, "rank": 10},
    "Northwestern":           {"gre": 320, "cgpa": 8.8, "toefl": 107, "acceptance_rate": 13, "research_req": False, "rank": 8},
    "Duke":                   {"gre": 318, "cgpa": 8.7, "toefl": 105, "acceptance_rate": 15, "research_req": False, "rank": 9},
    "Yale":                   {"gre": 324, "cgpa": 9.1, "toefl": 110, "acceptance_rate": 8,  "research_req": True,  "rank": 5},
    "UPenn":                  {"gre": 319, "cgpa": 8.7, "toefl": 106, "acceptance_rate": 14, "research_req": False, "rank": 8},
    "Rice":                   {"gre": 316, "cgpa": 8.5, "toefl": 103, "acceptance_rate": 18, "research_req": False, "rank": 10},
    "UIUC":                   {"gre": 317, "cgpa": 8.5, "toefl": 103, "acceptance_rate": 20, "research_req": False, "rank": 10},
    "University of Washington": {"gre": 316, "cgpa": 8.4, "toefl": 102, "acceptance_rate": 22, "research_req": False, "rank": 11},
    # ── Moderately Selective ──
    "Purdue":                 {"gre": 314, "cgpa": 8.2, "toefl": 100, "acceptance_rate": 28, "research_req": False, "rank": 12},
    "Texas A&M":              {"gre": 312, "cgpa": 8.1, "toefl": 99,  "acceptance_rate": 32, "research_req": False, "rank": 15},
    "Penn State":             {"gre": 311, "cgpa": 8.0, "toefl": 98,  "acceptance_rate": 33, "research_req": False, "rank": 16},
    "Ohio State":             {"gre": 310, "cgpa": 8.0, "toefl": 97,  "acceptance_rate": 35, "research_req": False, "rank": 17},
    "University of Wisconsin": {"gre": 313, "cgpa": 8.2, "toefl": 100, "acceptance_rate": 30, "research_req": False, "rank": 14},
    "University of Minnesota": {"gre": 311, "cgpa": 8.0, "toefl": 98,  "acceptance_rate": 34, "research_req": False, "rank": 16},
    "Virginia Tech":          {"gre": 309, "cgpa": 7.9, "toefl": 97,  "acceptance_rate": 38, "research_req": False, "rank": 20},
    "NC State":               {"gre": 308, "cgpa": 7.8, "toefl": 96,  "acceptance_rate": 40, "research_req": False, "rank": 22},
    "University of Florida":  {"gre": 309, "cgpa": 7.9, "toefl": 96,  "acceptance_rate": 38, "research_req": False, "rank": 21},
    "Northeastern":           {"gre": 310, "cgpa": 8.0, "toefl": 98,  "acceptance_rate": 35, "research_req": False, "rank": 18},
    "Boston University":      {"gre": 308, "cgpa": 7.9, "toefl": 96,  "acceptance_rate": 40, "research_req": False, "rank": 22},
    "Rutgers":                {"gre": 307, "cgpa": 7.8, "toefl": 95,  "acceptance_rate": 42, "research_req": False, "rank": 24},
    "UT Austin":              {"gre": 315, "cgpa": 8.3, "toefl": 100, "acceptance_rate": 25, "research_req": False, "rank": 12},
    # ── Less Selective ──
    "ASU":                    {"gre": 307, "cgpa": 7.8, "toefl": 95,  "acceptance_rate": 48, "research_req": False, "rank": 25},
    "UT Dallas":              {"gre": 308, "cgpa": 7.9, "toefl": 96,  "acceptance_rate": 45, "research_req": False, "rank": 23},
    "Stevens Institute":      {"gre": 306, "cgpa": 7.7, "toefl": 94,  "acceptance_rate": 48, "research_req": False, "rank": 26},
    "Drexel":                 {"gre": 305, "cgpa": 7.7, "toefl": 93,  "acceptance_rate": 50, "research_req": False, "rank": 27},
    "University of Colorado": {"gre": 308, "cgpa": 7.8, "toefl": 95,  "acceptance_rate": 45, "research_req": False, "rank": 24},
    "University of Arizona":  {"gre": 306, "cgpa": 7.7, "toefl": 93,  "acceptance_rate": 50, "research_req": False, "rank": 27},
    "SUNY Stony Brook":       {"gre": 307, "cgpa": 7.8, "toefl": 94,  "acceptance_rate": 47, "research_req": False, "rank": 25},
    "Indiana University":     {"gre": 306, "cgpa": 7.7, "toefl": 93,  "acceptance_rate": 52, "research_req": False, "rank": 28},
    "SUNY Buffalo":           {"gre": 304, "cgpa": 7.5, "toefl": 92,  "acceptance_rate": 58, "research_req": False, "rank": 32},
    "Wayne State":            {"gre": 302, "cgpa": 7.3, "toefl": 90,  "acceptance_rate": 62, "research_req": False, "rank": 36},
    "UMass Amherst":          {"gre": 309, "cgpa": 7.9, "toefl": 96,  "acceptance_rate": 40, "research_req": False, "rank": 21},
    "UMass Lowell":           {"gre": 303, "cgpa": 7.4, "toefl": 91,  "acceptance_rate": 60, "research_req": False, "rank": 34},
    "DePaul":                 {"gre": 300, "cgpa": 7.2, "toefl": 89,  "acceptance_rate": 65, "research_req": False, "rank": 38},
    "Pace University":        {"gre": 298, "cgpa": 7.0, "toefl": 87,  "acceptance_rate": 70, "research_req": False, "rank": 42},
    "Wichita State":          {"gre": 295, "cgpa": 6.8, "toefl": 84,  "acceptance_rate": 78, "research_req": False, "rank": 50},
    "Harrisburg University":  {"gre": 293, "cgpa": 6.6, "toefl": 82,  "acceptance_rate": 82, "research_req": False, "rank": 55},
}

def compare_university(uni_name: str, gre: float, cgpa: float, toefl: float, research: int) -> dict | None:
    key = None
    for k in UNIVERSITY_PROFILES:
        if k.lower() in uni_name.lower() or uni_name.lower() in k.lower():
            key = k
            break
    if not key:
        return None

    uni = UNIVERSITY_PROFILES[key]
    gre_gap   = gre   - uni["gre"]
    cgpa_gap  = round(cgpa - uni["cgpa"], 2)
    toefl_gap = toefl - uni["toefl"]

    gre_strength   = min(100, max(0, int((gre   / uni["gre"])   * 80)))
    cgpa_strength  = min(100, max(0, int((cgpa  / uni["cgpa"])  * 80)))
    toefl_strength = min(100, max(0, int((toefl / uni["toefl"]) * 80)))

    avg_gap = (gre_gap + cgpa_gap * 5 + toefl_gap * 0.5) / 3
    if avg_gap >= 0 and (research == 1 or not uni["research_req"]):
        verdict = ("Strong Match ✅", "#5cb85c")
    elif avg_gap >= -5:
        verdict = ("Borderline — Apply ⚡", "#c9a84c")
    elif avg_gap >= -12:
        verdict = ("Reach School 🎯", "#e8963a")
    else:
        verdict = ("Significant Gap ⚠️", "#d9534f")

    return {
        "name": key,
        "rank": uni["rank"],
        "acceptance_rate": uni["acceptance_rate"],
        "research_req": uni["research_req"],
        "gre_gap": gre_gap,
        "cgpa_gap": cgpa_gap,
        "toefl_gap": toefl_gap,
        "gre_avg": uni["gre"],
        "cgpa_avg": uni["cgpa"],
        "toefl_avg": uni["toefl"],
        "gre_strength": gre_strength,
        "cgpa_strength": cgpa_strength,
        "toefl_strength": toefl_strength,
        "verdict": verdict[0],
        "verdict_colour": verdict[1],
    }

# ═══════════════════════════════════════════════════════════
# ── NEW FEATURE 3: IMPROVEMENT ROADMAP GENERATOR ──
# ═══════════════════════════════════════════════════════════
def generate_roadmap(gre: float, cgpa: float, toefl: float, sop_lor: float, research: int) -> list[dict]:
    weeks = []
    week = 1

    gre_weak   = gre   < 310
    toefl_weak = toefl < 100
    sop_weak   = sop_lor < 3
    res_weak   = research == 0

    if gre_weak:
        weeks.append({
            "range": f"Week {week}–{week+2}",
            "focus": "GRE Quantitative Prep",
            "icon": "📐",
            "tasks": [
                "Complete Khan Academy Algebra & Arithmetic refresher",
                "Work through Manhattan Prep GRE Quant Strategy Guide (Ch 1–6)",
                "Do 30 practice problems daily on Magoosh or PowerPrep",
            ],
            "goal": f"Raise GRE from {int(gre)} → {min(int(gre)+15, 340)} (target 310+)",
            "colour": "#d9534f"
        })
        week += 3
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "GRE Verbal & Full Practice Tests",
            "icon": "📚",
            "tasks": [
                "Learn 10 high-frequency GRE vocab words per day",
                "Take 2 full-length timed practice tests",
                "Review all wrong answers — identify patterns",
            ],
            "goal": "Lock in score gains before test day",
            "colour": "#e8963a"
        })
        week += 2

    if toefl_weak:
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "TOEFL Score Improvement",
            "icon": "🗣",
            "tasks": [
                "Focus on Writing section: practice integrated + independent tasks daily",
                "Use ETS Official TOEFL Practice Online (TPO) tests",
                "Record yourself speaking for 30 min/day and review for fluency",
            ],
            "goal": f"Push TOEFL from {int(toefl)} → 100+",
            "colour": "#d9534f"
        })
        week += 2

    if res_weak:
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "Build Research Experience",
            "icon": "🔬",
            "tasks": [
                "Email 5 professors at your university requesting research assistance",
                "Start a self-directed project: pick a dataset on Kaggle, define a research question",
                "Set up a GitHub repo and document your methodology clearly",
            ],
            "goal": "Have at least 1 research project to mention in SOP",
            "colour": "#4a90d9"
        })
        week += 2

    if sop_weak:
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "SOP Drafting & LOR Planning",
            "icon": "✍️",
            "tasks": [
                "Write a 500-word rough SOP draft — focus on your 'why grad school' story",
                "Identify 3 recommenders and send briefing materials to each",
                "Research 3 specific faculty members at target schools to name in your SOPs",
            ],
            "goal": "First draft SOP complete; LOR requests sent",
            "colour": "#c9a84c"
        })
        week += 2
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "SOP Refinement & School Research",
            "icon": "🎯",
            "tasks": [
                "Get SOP feedback from a mentor, professor, or writing center",
                "Tailor SOP for your top 3 schools (name faculty, align with their research)",
                "Finalize school list: 4 reach, 4 match, 3 safe schools",
            ],
            "goal": "Polished, school-specific SOPs ready",
            "colour": "#c9a84c"
        })
        week += 2
    else:
        weeks.append({
            "range": f"Week {week}–{week+1}",
            "focus": "SOP Polish & School Targeting",
            "icon": "✍️",
            "tasks": [
                "Tailor your SOP for each target school — name specific faculty",
                "Get feedback on SOP from a professor or mentor",
                "Finalize your school list (10–12 total across all tiers)",
            ],
            "goal": "School-specific, polished SOPs ready",
            "colour": "#c9a84c"
        })
        week += 2

    weeks.append({
        "range": f"Week {week}–{week+1}",
        "focus": "Application Submission Sprint",
        "icon": "🚀",
        "tasks": [
            "Complete all online application forms — double-check every field",
            "Upload transcripts, CV, and any supporting documents",
            "Confirm LOR submissions with all recommenders (follow up if needed)",
            "Submit applications 3–5 days before each deadline",
        ],
        "goal": "All applications submitted on time",
        "colour": "#5cb85c"
    })

    return weeks

# ═══════════════════════════════════════════════════════════
# ── NEW FEATURE 4: PDF REPORT GENERATOR ──
# ═══════════════════════════════════════════════════════════
def generate_pdf_report(username: str, prediction: dict, academic: dict, checklist_data: list | None = None) -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.85*inch, rightMargin=0.85*inch,
                             topMargin=0.85*inch, bottomMargin=0.85*inch)

    GOLD  = colors.HexColor("#c9a84c")
    DARK  = colors.HexColor("#0d0f14")
    PANEL = colors.HexColor("#181c26")
    PANEL2= colors.HexColor("#1c2030")
    MUTED = colors.HexColor("#6b7080")
    GREEN = colors.HexColor("#5cb85c")
    RED   = colors.HexColor("#d9534f")
    AMBER = colors.HexColor("#e8963a")
    BLUE  = colors.HexColor("#4a90d9")
    LIGHT = colors.HexColor("#e8e4dc")
    ACC   = colors.HexColor("#1a1f2e")
    BORDER= colors.HexColor("#252834")

    base = getSampleStyleSheet()["Normal"]

    def S(name, **kw):
        return ParagraphStyle(name, parent=base, **kw)

    h1    = S("h1", fontSize=26, textColor=GOLD, fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
    sub   = S("sub", fontSize=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=2)
    h2    = S("h2", fontSize=14, textColor=LIGHT, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    body  = S("body", fontSize=9.5, textColor=LIGHT, leading=14, spaceAfter=3)
    small = S("small", fontSize=8.5, textColor=MUTED, leading=13)
    tip   = S("tip", fontSize=9, textColor=colors.HexColor("#c8b87a"), leading=13, spaceAfter=3)
    lbl   = S("lbl", fontSize=8.5, textColor=MUTED, fontName="Helvetica-Bold")
    val   = S("val", fontSize=10, textColor=LIGHT, fontName="Helvetica-Bold")
    foot  = S("foot", fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)

    story = []
    W = 6.3 * inch

    # ── Header ──
    story += [
        Spacer(1, 0.1*inch),
        Paragraph("GradPath AI", h1),
        Paragraph("Graduate Admission Intelligence Report", sub),
        Paragraph(f"Prepared for: {username}  ·  {datetime.date.today().strftime('%B %d, %Y')}", sub),
        HRFlowable(width=W, thickness=1.5, color=GOLD, spaceAfter=12, spaceBefore=6),
    ]

    # ── Score banner ──
    pct  = prediction["probability"]
    band = prediction["band"]
    bc   = {"High": GREEN, "Moderate": GOLD, "Low-Moderate": AMBER, "Low": RED}.get(band, GOLD)

    banner = Table([[
        Paragraph(f"{pct}%", S("pct", fontSize=36, textColor=GOLD, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        Paragraph(f"<b>{band} Chance</b><br/>Predicted Acceptance Probability",
                  S("bnd", fontSize=11, textColor=bc, leading=16)),
    ]], colWidths=[2*inch, 4.3*inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), PANEL),
        ("BOX",          (0,0),(-1,-1), 1.5, GOLD),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 16),
        ("RIGHTPADDING", (0,0),(-1,-1), 16),
        ("TOPPADDING",   (0,0),(-1,-1), 14),
        ("BOTTOMPADDING",(0,0),(-1,-1), 14),
    ]))
    story += [banner, Spacer(1, 12)]

    # ── Academic profile ──
    story.append(Paragraph("Academic Profile", h2))
    fields = [
        ("GRE Score",        str(academic.get("gre",  "—"))),
        ("CGPA",             f"{academic.get('cgpa','—')} / 10"),
        ("TOEFL Score",      str(academic.get("toefl","—"))),
        ("SOP / LOR Rating", f"{academic.get('sop_lor','—')} / 5"),
        ("Research",         "Yes" if academic.get("research", 0) else "No"),
        ("Uni Rating",       f"{academic.get('uni_rating','—')} / 5"),
    ]
    pt = Table([[Paragraph(l, lbl), Paragraph(v, val)] for l, v in fields],
               colWidths=[W*0.38, W*0.62])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), PANEL),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [PANEL, PANEL2]),
        ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    story += [pt, Spacer(1, 12)]

    # ── Factor breakdown ──
    story.append(Paragraph("Factor Breakdown", h2))
    frows = [[Paragraph("<b>Factor</b>", lbl), Paragraph("<b>Score</b>", lbl), Paragraph("<b>Status</b>", lbl)]]
    for factor, score in prediction["factors"].items():
        sc = GREEN if score >= 70 else GOLD if score >= 50 else RED
        st_txt = "Strong" if score >= 70 else "Moderate" if score >= 50 else "Weak"
        frows.append([
            Paragraph(factor, body),
            Paragraph(f"{score}%", S("fs", fontSize=10, textColor=sc, fontName="Helvetica-Bold")),
            Paragraph(st_txt,      S("st", fontSize=9,  textColor=sc)),
        ])
    ft = Table(frows, colWidths=[W*0.4, W*0.25, W*0.35])
    ft.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  ACC),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [PANEL, PANEL2]),
        ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("TEXTCOLOR",     (0,0),(-1,0),  GOLD),
    ]))
    story += [ft, Spacer(1, 12)]

    # ── Tips ──
    if prediction.get("tips"):
        story.append(Paragraph("Priority Improvements", h2))
        for t in prediction["tips"]:
            story.append(Paragraph(f"   {t}", tip))
        story.append(Spacer(1, 8))

    # ── Checklist ──
    if checklist_data:
        story.append(PageBreak())
        story.append(Paragraph("Application Checklist", h2))
        story.append(Paragraph("Track your progress across all target schools.", small))
        story.append(Spacer(1, 8))
        for school in checklist_data:
            name     = school.get("name", "")
            deadline = school.get("deadline", "—")
            tasks    = school.get("tasks", {})
            done     = sum(1 for v in tasks.values() if v)
            total    = len(tasks)
            pct_done = int((done / total) * 100) if total else 0
            story.append(Paragraph(
                f"<b>{name}</b>  ·  Deadline: {deadline}  ·  Progress: {done}/{total} ({pct_done}%)",
                S("ch", fontSize=10, textColor=LIGHT, spaceBefore=6, spaceAfter=3)
            ))
            for task_name, done_flag in tasks.items():
                mark = "✓" if done_flag else "○"
                tc   = GREEN if done_flag else MUTED
                story.append(Paragraph(
                    f"   {mark}  {task_name}",
                    S("tk", fontSize=9, textColor=LIGHT if done_flag else MUTED, spaceAfter=2)
                ))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=4))

    # ── Roadmap ──
    story.append(PageBreak())
    story.append(Paragraph("Personalized Improvement Roadmap", h2))
    roadmap = generate_roadmap(
        academic.get("gre", 310), academic.get("cgpa", 7.5),
        academic.get("toefl", 95), academic.get("sop_lor", 3),
        academic.get("research", 0)
    )
    for phase in roadmap:
        ph_tbl = Table(
            [[Paragraph(f"{phase['icon']}  {phase['range']}  —  {phase['focus']}",
                        S("ph", fontSize=10, textColor=GOLD, fontName="Helvetica-Bold"))]],
            colWidths=[W]
        )
        ph_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), ACC),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#2e3650")),
        ]))
        story.append(ph_tbl)
        for task in phase["tasks"]:
            story.append(Paragraph(f"   •  {task}", body))
        story.append(Paragraph(f"Goal: {phase['goal']}", small))
        story.append(Spacer(1, 6))

    # ── Footer ──
    story += [
        Spacer(1, 0.3*inch),
        HRFlowable(width=W, thickness=0.5, color=MUTED, spaceAfter=6),
        Paragraph("GradPath AI  ·  Predictions are probabilistic, not deterministic  ·  Not affiliated with any university", foot),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<h2 style="font-family:\'Playfair Display\',serif;color:#c9a84c;margin-bottom:0">GradPath</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7080;font-size:0.8rem;margin-top:0">Graduate Admission Intelligence</p>', unsafe_allow_html=True)
    st.divider()

    # ── Auth ──
    if not st.session_state.user:
        tab_choice = st.radio("Account", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        st.session_state.auth_tab = tab_choice

        uname = st.text_input("Username", key="auth_uname")
        pw    = st.text_input("Password", type="password", key="auth_pw")

        if tab_choice == "Login":
            if st.button("Login →", use_container_width=True):
                ok, msg = login(uname, pw)
                if ok:
                    st.session_state.user = uname
                    st.rerun()
                else:
                    st.error(msg)
        else:
            pw2 = st.text_input("Confirm Password", type="password", key="auth_pw2")
            if st.button("Create Account →", use_container_width=True):
                if pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(uname, pw)
                    if ok:
                        st.session_state.user = uname
                        st.rerun()
                    else:
                        st.error(msg)

    else:
        st.markdown(f'<p class="muted">Signed in as</p><b style="color:#e8e4dc">{st.session_state.user}</b>', unsafe_allow_html=True)

        if st.button("Sign Out", use_container_width=True):
            logout()
            st.rerun()

        st.divider()

        # ── Chat history ──
        st.markdown('<p class="muted" style="margin-bottom:0.4rem">SAVED CHATS</p>', unsafe_allow_html=True)
        user_chats = chats.get(st.session_state.user, {})
        if user_chats:
            for cid in sorted(user_chats.keys(), reverse=True)[:8]:
                label = cid[:16].replace("T", "  ").replace("-", "/")[:13]
                if st.button(f"🗂 {label}", use_container_width=True, key=f"load_{cid}"):
                    load_chat(cid)
                    st.rerun()
        else:
            st.markdown('<p class="muted">No saved chats yet.</p>', unsafe_allow_html=True)

        if st.button("＋ New Chat", use_container_width=True):
            new_chat()
            st.rerun()

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════
if not st.session_state.user:
    # ── Landing splash ──
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem">
        <h1 style="font-family:'Playfair Display',serif;font-size:3.2rem;color:#e8e4dc;margin-bottom:0.3rem">
            GradPath <span style="color:#c9a84c">AI</span>
        </h1>
        <p style="color:#6b7080;font-size:1.1rem;max-width:520px;margin:0 auto 2rem">
            Predict your graduate school admission chances with precision. 
            Upload your profile, get your score, then ask anything.
        </p>
        <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap">
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">📊</div>
                <b>Data-Driven</b>
                <p class="muted">Weighted model across 6 admission factors</p>
            </div>
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">💬</div>
                <b>AI Chat</b>
                <p class="muted">Ask follow-up questions on your results</p>
            </div>
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">🔒</div>
                <b>Secure Profiles</b>
                <p class="muted">Password-protected with saved chat history</p>
            </div>
        </div>
        <p style="color:#6b7080;margin-top:2rem;font-size:0.9rem">← Log in or register from the sidebar</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PROFILE + PREDICTION PANEL
# ─────────────────────────────────────────────
prefs = profiles[st.session_state.user].get("academic", {})

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<h2 style="color:#e8e4dc">Academic Profile</h2>', unsafe_allow_html=True)

    with st.form("profile_form"):
        gre = st.number_input("GRE Score (260–340)", min_value=260, max_value=340, value=int(prefs.get("gre", 310)), step=1)

        cgpa = st.number_input("CGPA (0.0–10.0)", min_value=0.0, max_value=10.0, value=float(prefs.get("cgpa", 7.5)), step=0.1, format="%.1f")

        toefl = st.number_input("TOEFL Score (0–120)", min_value=0, max_value=120, value=int(prefs.get("toefl", 95)), step=1)

        st.markdown('<p style="color:#6b7080;font-size:0.82rem;margin-bottom:-0.6rem">SOP / LOR Combined Rating</p>', unsafe_allow_html=True)
        sop_lor = st.slider("", 1, 5, int(prefs.get("sop_lor", 3)), label_visibility="collapsed", key="sop_lor_slider")

        research = st.radio("Research Experience", [0, 1], index=int(prefs.get("research", 0)),
                            format_func=lambda x: "Yes" if x else "No", horizontal=True)

        st.markdown('<p style="color:#6b7080;font-size:0.82rem;margin-bottom:-0.6rem">Target University Rating</p>', unsafe_allow_html=True)
        uni_rating = st.slider("", 1, 5, int(prefs.get("uni_rating", 3)), label_visibility="collapsed", key="uni_rating_slider")

        submitted = st.form_submit_button("Run Prediction →", use_container_width=True)

    # ── Score Guide expander — separate from the form ──
    with st.expander("📖 What do these scores mean?", expanded=False):

        st.markdown("**📝 GRE Score** *(range: 260–340)*")
        st.markdown(
            "The Graduate Record Examination — total of Verbal + Quantitative sections.\n\n"
            "- 🔴 **260–299** = weak\n"
            "- 🟠 **300–314** = average\n"
            "- 🟡 **315–329** = competitive\n"
            "- 🟢 **330–340** = exceptional\n\n"
            "Most MS programs expect **310+**. Top programs like MIT and Stanford want **320+**."
        )
        st.divider()

        st.markdown("**🎓 CGPA** *(range: 0.0–10.0)*")
        st.markdown(
            "Your Cumulative Grade Point Average on a **10-point scale** (standard in India and many countries).\n\n"
            "On a 4.0 scale? Multiply by 2.5 to approximate — e.g. 3.5 GPA ≈ 8.75 CGPA.\n\n"
            "- 🔴 **Below 7.0** = weak\n"
            "- 🟠 **7.0–7.9** = average\n"
            "- 🟡 **8.0–8.9** = strong\n"
            "- 🟢 **9.0–10.0** = exceptional"
        )
        st.divider()

        st.markdown("**🗣 TOEFL Score** *(range: 0–120)*")
        st.markdown(
            "Test of English as a Foreign Language — required for most non-native English speakers.\n\n"
            "- 🔴 **Below 90** = below minimum for most programs\n"
            "- 🟠 **90–99** = meets minimum cutoff\n"
            "- 🟡 **100–109** = competitive\n"
            "- 🟢 **110–120** = excellent\n\n"
            "Native English speakers or students from English-medium universities may be exempt — enter **110** if your TOEFL was waived."
        )
        st.divider()

        st.markdown("**✍️ SOP / LOR Combined Rating** *(1 = weak · 5 = exceptional)*")
        st.markdown(
            "A self-assessment of your **Statement of Purpose** (motivation essay) and **Letters of Recommendation** (references from professors or supervisors).\n\n"
            "- 🔴 **1** = Generic drafts, unknown recommenders\n"
            "- 🟠 **2** = Basic content, no standout moments\n"
            "- 🟡 **3** = Solid, clear narrative\n"
            "- 🟡 **4** = Strong, tailored to each school, senior recommenders\n"
            "- 🟢 **5** = Outstanding SOP + research supervisor letters"
        )
        st.divider()

        st.markdown("**🔬 Research Experience**")
        st.markdown(
            "Select **Yes** if you have at least one substantial research experience — a thesis, lab project, "
            "internship with a research output, conference paper, or publication.\n\n"
            "Research experience can add **+8–15%** to your predicted acceptance chance for competitive programs."
        )
        st.divider()

        st.markdown("**🏛 Target University Rating** *(1 = easy admit · 5 = elite)*")
        st.markdown(
            "How selective are the universities you are primarily targeting?\n\n"
            "- 🟢 **1** = Open / rolling admission — acceptance rate 60%+\n"
            "- 🟡 **2** = Less selective — e.g. SUNY Buffalo, Wayne State (40–60%)\n"
            "- 🟡 **3** = Moderate — e.g. Northeastern, ASU, Texas A&M (20–40%)\n"
            "- 🟠 **4** = Highly selective — e.g. Georgia Tech, UCLA, Purdue (8–20%)\n"
            "- 🔴 **5** = Elite — e.g. MIT, Stanford, CMU, UC Berkeley (below 8%)"
        )

    if submitted:
        result = predict_admission(gre, cgpa, toefl, sop_lor, research, uni_rating)
        st.session_state.last_prediction = result

        # Save profile
        profiles[st.session_state.user]["academic"] = {
            "gre": gre, "cgpa": cgpa, "toefl": toefl,
            "sop_lor": sop_lor, "research": research, "uni_rating": uni_rating
        }
        save_json(PROFILE_FILE, profiles)

        # Inject prediction summary into chat
        pct   = result["probability"]
        band  = result["band"]
        tips  = result["tips"]
        tip_lines = "\n".join(f"  • {t}" for t in tips) if tips else "  • Profile looks competitive!"

        summary = (
            f"📊 **New prediction run** — {pct}% ({band} chance)\n\n"
            f"**Inputs:** GRE {gre} | CGPA {cgpa} | TOEFL {toefl} | SOP/LOR {sop_lor}/5 | "
            f"Research: {'Yes' if research else 'No'} | Uni Rating: {uni_rating}/5\n\n"
            f"**Areas to improve:**\n{tip_lines}\n\n"
            "Feel free to ask me anything about these results!"
        )

        st.session_state.current_chat.append({"role": "assistant", "content": summary})
        save_chat()
        st.rerun()

with col_right:
    st.markdown('<h2 style="color:#e8e4dc">Prediction Results</h2>', unsafe_allow_html=True)

    pred = st.session_state.last_prediction

    if not pred:
        # Try to recover from profile if previously saved
        if prefs:
            pred = predict_admission(
                prefs.get("gre",310), prefs.get("cgpa",7.5), prefs.get("toefl",95),
                prefs.get("sop_lor",3), prefs.get("research",0), prefs.get("uni_rating",3)
            )

    if pred:
        pct    = pred["probability"]
        band   = pred["band"]
        colour = pred["colour"]
        tips   = pred["tips"]

        # ── Big score ──
        st.markdown(f"""
        <div class="grad-card-accent" style="text-align:center">
            <p class="muted" style="margin-bottom:0.3rem">Predicted Acceptance Probability</p>
            <span class="score-badge">{pct}%</span>
            <p style="margin-top:0.6rem;color:{colour};font-weight:600;font-size:1rem">{band} Chance</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Factor breakdown ──
        st.markdown('<div class="grad-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:600;margin-bottom:0.8rem">Factor Breakdown</p>', unsafe_allow_html=True)
        for factor, score in pred["factors"].items():
            bar_w = int(score)
            bar_c = "#5cb85c" if score >= 70 else "#c9a84c" if score >= 50 else "#d9534f"
            st.markdown(f"""
            <div style="margin-bottom:0.65rem">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                    <span>{factor}</span><span style="color:{bar_c};font-weight:600">{score}%</span>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{bar_w}%;background:{bar_c}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Tips ──
        if tips:
            st.markdown('<div class="warn-box">⚡ <b>Priority Improvements</b><br>' +
                        "<br>".join(f"• {t}" for t in tips) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ Your profile is strong across all factors.</div>', unsafe_allow_html=True)

        st.markdown('<p class="muted">⚠ Predictions are probabilistic. Actual decisions depend on essays, interviews, and competition.</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="grad-card" style="text-align:center;padding:3rem">
            <div style="font-size:2rem">🎓</div>
            <p style="color:#6b7080">Fill in your academic profile on the left and hit <b>Run Prediction</b> to see your results here.</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT SECTION
# ─────────────────────────────────────────────
st.divider()
st.markdown('<h2 style="color:#e8e4dc">Ask GradPath AI</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Ask follow-up questions about your results — GRE strategy, SOP tips, school lists, and more.</p>', unsafe_allow_html=True)

# Display messages
for msg in st.session_state.current_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("e.g. How can I improve my chances? / Should I retake the GRE?")

if user_input:
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    acad = profiles[st.session_state.user].get("academic", {})
    resp = ai_response(user_input, st.session_state.last_prediction, acad)

    st.session_state.current_chat.append({"role": "assistant", "content": resp})
    save_chat()
    st.rerun()

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
if st.session_state.current_chat:
    st.divider()
    export_data = json.dumps({
        "user":       st.session_state.user,
        "exported":   str(datetime.datetime.now()),
        "prediction": st.session_state.last_prediction,
        "chat":       st.session_state.current_chat
    }, indent=4)
    st.download_button(
        "📥 Export Session (JSON)",
        export_data,
        file_name=f"gradpath_{st.session_state.user}_{datetime.date.today()}.json",
        mime="application/json"
    )

# ═══════════════════════════════════════════════════════════
# TOOLS SECTION — tabbed navigation
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown('<h2 style="color:#e8e4dc">🛠 Planning Tools</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Everything you need to plan, track, and improve your applications — all in one place.</p>', unsafe_allow_html=True)

tool_tab1, tool_tab2, tool_tab3 = st.tabs([
    "🔍  University Comparison",
    "✅  Application Checklist",
    "🗺  Improvement Roadmap",
])

# ───────────────────────────────────────────
# TAB 1 — UNIVERSITY COMPARISON
# ───────────────────────────────────────────
with tool_tab1:
    st.markdown("#### Compare Your Profile to Any University")
    st.caption("Enter up to 4 university names and see exactly how your scores compare to their average admitted student.")

    acad_now = profiles[st.session_state.user].get("academic", {})
    compare_gre   = acad_now.get("gre",      310)
    compare_cgpa  = acad_now.get("cgpa",     7.5)
    compare_toefl = acad_now.get("toefl",    95)
    compare_res   = acad_now.get("research", 0)

    uni_input_col1, uni_input_col2 = st.columns([3, 1])
    with uni_input_col1:
        uni_names_raw = st.text_input(
            "University names (comma-separated)",
            placeholder="e.g. MIT, Georgia Tech, Purdue, ASU",
            key="uni_compare_input"
        )
    with uni_input_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_compare = st.button("Compare →", key="run_compare_btn", use_container_width=True)

    st.caption("Available: MIT, Stanford, Carnegie Mellon, UC Berkeley, Caltech, Princeton, Harvard, Yale, Johns Hopkins, Cornell, Northwestern, Duke, UPenn, Rice, UCLA, Columbia, University of Michigan, Georgia Tech, UIUC, University of Washington, NYU, UC San Diego, UT Austin, Purdue, Texas A&M, Penn State, Ohio State, University of Wisconsin, University of Minnesota, Virginia Tech, NC State, University of Florida, Northeastern, Boston University, Rutgers, UMass Amherst, ASU, UT Dallas, Stevens Institute, Drexel, University of Colorado, University of Arizona, SUNY Stony Brook, Indiana University, SUNY Buffalo, Wayne State, UMass Lowell, DePaul, Pace University, Wichita State, Harrisburg University")

    if run_compare and uni_names_raw.strip():
        names   = [n.strip() for n in uni_names_raw.split(",") if n.strip()][:4]
        results = [compare_university(n, compare_gre, compare_cgpa, compare_toefl, compare_res) for n in names]
        found   = [(names[i], r) for i, r in enumerate(results) if r is not None]
        missing = [names[i] for i, r in enumerate(results) if r is None]

        if missing:
            st.warning(f"Not found in database: {', '.join(missing)}")

        if found:
            compare_cols = st.columns(len(found))
            for col, (orig_name, r) in zip(compare_cols, found):
                with col:
                    def gap_label(gap, unit=""):
                        prefix = "+" if gap > 0 else ""
                        return f"{prefix}{gap}{unit}"

                    st.markdown(f"### 🏛 {r['name']}")
                    st.caption(f"Rank ~#{r['rank']}  ·  Acceptance rate: {r['acceptance_rate']}%{'  ·  Research required' if r['research_req'] else ''}")
                    st.markdown(f"**{r['verdict']}**")
                    st.divider()
                    st.metric(label=f"GRE  (avg: {r['gre_avg']})",    value=int(compare_gre),  delta=gap_label(r["gre_gap"],   " pts"))
                    st.metric(label=f"CGPA  (avg: {r['cgpa_avg']})",   value=compare_cgpa,       delta=gap_label(r["cgpa_gap"]))
                    st.metric(label=f"TOEFL  (avg: {r['toefl_avg']})", value=int(compare_toefl), delta=gap_label(r["toefl_gap"], " pts"))

# ───────────────────────────────────────────
# TAB 2 — APPLICATION CHECKLIST
# ───────────────────────────────────────────
with tool_tab2:
    st.markdown("#### Application Checklist & Deadline Tracker")
    st.caption("Add every school you are applying to, set a deadline, and tick off tasks as you complete them.")

    user_cl = checklists.get(st.session_state.user, [])
    DEFAULT_TASKS = ["SOP drafted", "SOP finalised", "LOR requested", "LOR submitted", "Transcripts ready", "Application form filled", "Application submitted"]

    with st.expander("➕ Add a School", expanded=len(user_cl) == 0):
        cl_col1, cl_col2, cl_col3 = st.columns([2, 1.5, 1])
        with cl_col1:
            new_school_name = st.text_input("School name", placeholder="e.g. Georgia Tech (CS)", key="cl_school_name")
        with cl_col2:
            new_school_deadline = st.date_input("Application deadline", value=None, min_value=datetime.date.today(), key="cl_deadline", format="MM/DD/YYYY")
        with cl_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            add_school_btn = st.button("Add School", key="cl_add_btn", use_container_width=True)

        if add_school_btn and new_school_name.strip():
            deadline_str = new_school_deadline.strftime("%b %d, %Y") if new_school_deadline else "—"
            entry = {
                "name":     new_school_name.strip(),
                "deadline": deadline_str,
                "tasks":    {t: False for t in DEFAULT_TASKS}
            }
            user_cl.append(entry)
            checklists[st.session_state.user] = user_cl
            save_json(CHECKLIST_FILE, checklists)
            st.rerun()

    if not user_cl:
        st.info("No schools added yet. Use the form above to start tracking your applications.")
    else:
        for idx, school in enumerate(user_cl):
            tasks    = school.get("tasks", {})
            done     = sum(1 for v in tasks.values() if v)
            total    = len(tasks)
            pct_done = int((done / total) * 100) if total else 0

            with st.expander(f"🏛 {school['name']}  ·  Deadline: {school['deadline']}  ·  {done}/{total} done", expanded=True):
                st.progress(pct_done / 100, text=f"{pct_done}% complete")

                task_cols = st.columns(2)
                changed = False
                for ti, (task_name, done_flag) in enumerate(tasks.items()):
                    with task_cols[ti % 2]:
                        new_val = st.checkbox(task_name, value=done_flag, key=f"cl_{idx}_{ti}")
                        if new_val != done_flag:
                            user_cl[idx]["tasks"][task_name] = new_val
                            changed = True

                if changed:
                    checklists[st.session_state.user] = user_cl
                    save_json(CHECKLIST_FILE, checklists)
                    st.rerun()

                if st.button(f"🗑 Remove {school['name']}", key=f"cl_remove_{idx}"):
                    user_cl.pop(idx)
                    checklists[st.session_state.user] = user_cl
                    save_json(CHECKLIST_FILE, checklists)
                    st.rerun()

# ───────────────────────────────────────────
# TAB 3 — IMPROVEMENT ROADMAP
# ───────────────────────────────────────────
with tool_tab3:
    st.markdown("#### Your Personalized Week-by-Week Roadmap")
    st.caption("A study and prep plan built around your weakest factors — no generic advice.")

    roadmap_acad = profiles[st.session_state.user].get("academic", {})

    if not roadmap_acad:
        st.info("Run a prediction first to generate your personalized roadmap.")
    else:
        roadmap = generate_roadmap(
            roadmap_acad.get("gre",      310),
            roadmap_acad.get("cgpa",     7.5),
            roadmap_acad.get("toefl",    95),
            roadmap_acad.get("sop_lor",  3),
            roadmap_acad.get("research", 0),
        )

        for phase in roadmap:
            with st.expander(f"{phase['icon']}  {phase['range']}  —  {phase['focus']}", expanded=True):
                st.caption(f"🎯 Goal: {phase['goal']}")
                for task in phase["tasks"]:
                    st.markdown(f"- {task}")

# ═══════════════════════════════════════════════════════════
# ── NEW FEATURE 4: PDF REPORT DOWNLOAD ──
# ═══════════════════════════════════════════════════════════
st.divider()

st.markdown('<h2 style="color:#e8e4dc">📄 Download Your Full Report (PDF)</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Export a polished PDF summarising your prediction, factor breakdown, checklist, and improvement roadmap — share it with an advisor.</p>', unsafe_allow_html=True)

pdf_pred = st.session_state.last_prediction
pdf_acad = profiles[st.session_state.user].get("academic", {})

if not pdf_pred or not pdf_acad:
    st.markdown('<div class="warn-box">⚡ Run a prediction first to enable the PDF export.</div>', unsafe_allow_html=True)
else:
    if st.button("📄 Generate & Download PDF Report", key="pdf_generate_btn", use_container_width=False):
        with st.spinner("Generating your report…"):
            cl_for_pdf = checklists.get(st.session_state.user, [])
            pdf_bytes  = generate_pdf_report(
                st.session_state.user,
                pdf_pred,
                pdf_acad,
                cl_for_pdf if cl_for_pdf else None
            )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"gradpath_report_{st.session_state.user}_{datetime.date.today()}.pdf",
            mime="application/pdf",
            key="pdf_download_btn"
        )
        st.markdown('<div class="info-box">✅ Report ready! Click <b>Download PDF</b> above to save it.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# EXPLORE PROGRAMS, MAJORS & APPLICATION GUIDE
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown('<h2 style="color:#e8e4dc">🎓 Explore Programs, Majors & How to Apply</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Browse by interest area — find majors you might enjoy, what careers they lead to, and exactly how to apply.</p>', unsafe_allow_html=True)

PROGRAM_DATA = {
    "💻 Computer Science & AI": {
        "description": "One of the most in-demand graduate fields. Covers algorithms, systems, AI, machine learning, and software engineering.",
        "majors": ["MS Computer Science", "MS Artificial Intelligence", "MS Machine Learning", "MS Cybersecurity", "MS Human-Computer Interaction", "MS Software Engineering"],
        "careers": ["Software Engineer", "ML Engineer", "AI Researcher", "Data Scientist", "Security Engineer", "Product Manager (Tech)"],
        "top_schools": ["MIT", "Stanford", "Carnegie Mellon", "UC Berkeley", "Georgia Tech", "University of Michigan"],
        "avg_gre": "315–330",
        "avg_cgpa": "8.5–10 / 10",
        "application_tips": [
            "Highlight coding projects on GitHub — admissions committees look at your portfolio",
            "Research the faculty you want to work with and name them specifically in your SOP",
            "LeetCode / competitive programming experience is a plus for top programs",
            "A research paper or publication significantly boosts your profile",
            "Apply to 10–14 programs spread across reach, match, and safe tiers",
        ],
        "timeline": "Deadlines: Dec 1–Jan 15 for Fall intake. Results: Feb–April.",
    },
    "📊 Data Science & Analytics": {
        "description": "Combines statistics, programming, and domain knowledge to extract insights from data. Highly interdisciplinary.",
        "majors": ["MS Data Science", "MS Business Analytics", "MS Statistics", "MS Computational Social Science", "MS Biostatistics", "MS Financial Analytics"],
        "careers": ["Data Scientist", "Data Analyst", "Business Intelligence Analyst", "Quantitative Analyst", "Research Scientist", "AI Product Manager"],
        "top_schools": ["Columbia", "NYU", "UC San Diego", "Georgia Tech", "Purdue", "Carnegie Mellon"],
        "avg_gre": "310–325",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Show a Kaggle portfolio, personal projects, or GitHub repos with data analysis work",
            "Statistics and linear algebra coursework should be highlighted in your SOP",
            "Industry experience with data tools (SQL, Python, R) is a strong differentiator",
            "Some programs require a writing sample or research statement — prepare one early",
            "Target programs that match your focus: some are more business-oriented, others more technical",
        ],
        "timeline": "Deadlines: Nov 15–Jan 1 for Fall intake. Results: Feb–March.",
    },
    "⚡ Electrical & Computer Engineering": {
        "description": "Covers hardware, circuits, signal processing, embedded systems, robotics, and computer architecture.",
        "majors": ["MS Electrical Engineering", "MS Computer Engineering", "MS Robotics", "MS Signal Processing", "MS Power Systems", "MS VLSI Design"],
        "careers": ["Hardware Engineer", "Robotics Engineer", "Embedded Systems Engineer", "RF Engineer", "Power Systems Engineer", "Chip Designer"],
        "top_schools": ["MIT", "Stanford", "Caltech", "Georgia Tech", "Purdue", "University of Michigan"],
        "avg_gre": "315–330",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Lab experience and hardware projects are very highly valued — mention specifics",
            "Research assistantships are common in EE — email professors directly before applying",
            "GRE Quantitative score is weighted heavily — aim for 165+",
            "Internships at hardware companies (Intel, Qualcomm, NVIDIA) strengthen your profile significantly",
            "Include any patents, publications, or conference presentations in your application",
        ],
        "timeline": "Deadlines: Dec 1–Jan 15 for Fall intake. Results: Feb–April.",
    },
    "🧬 Biomedical & Life Sciences": {
        "description": "Applies engineering and computational methods to biology, medicine, and healthcare. Fast-growing field with high impact.",
        "majors": ["MS Biomedical Engineering", "MS Bioinformatics", "MS Computational Biology", "MS Neuroscience", "MS Genomics", "MS Health Informatics"],
        "careers": ["Biomedical Engineer", "Bioinformatics Scientist", "Clinical Data Analyst", "Research Scientist", "Pharmaceutical Analyst", "Healthcare Data Engineer"],
        "top_schools": ["Johns Hopkins", "UCSF", "MIT", "Stanford", "Georgia Tech", "University of Michigan"],
        "avg_gre": "308–322",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Wet lab or computational biology research experience is almost mandatory for top programs",
            "A clear research interest stated in your SOP matters more than in other fields",
            "Strong letters from research supervisors carry more weight than course instructors",
            "Some programs require GRE Biology Subject Test — check each school's requirements",
            "Highlight any exposure to bioinformatics tools (Python, R, BLAST, genome pipelines)",
        ],
        "timeline": "Deadlines: Nov 1–Dec 15 for Fall intake. Results: Jan–March.",
    },
    "💰 Financial Engineering & Quantitative Finance": {
        "description": "Applies mathematics and programming to financial markets, risk management, and investment strategies.",
        "majors": ["MS Financial Engineering", "MS Quantitative Finance", "MS Mathematical Finance", "MS Risk Management", "MS FinTech", "MS Computational Finance"],
        "careers": ["Quantitative Analyst", "Risk Manager", "Derivatives Trader", "Portfolio Manager", "FinTech Developer", "Actuarial Analyst"],
        "top_schools": ["Columbia", "NYU Courant", "Carnegie Mellon", "Princeton", "UC Berkeley", "Baruch College"],
        "avg_gre": "318–330",
        "avg_cgpa": "8.5–10 / 10",
        "application_tips": [
            "Strong mathematics background (calculus, probability, linear algebra) is essential — highlight it",
            "Programming skills in Python, C++, or MATLAB are expected — show projects",
            "Finance internships or CFA Level 1 significantly strengthen non-finance undergrad applicants",
            "Some programs require the GMAT instead of or in addition to the GRE — check each school",
            "Interview preparation is required for top MFE programs — practice stochastic calculus questions",
        ],
        "timeline": "Deadlines: Nov 1–Jan 1 for Fall intake. Results: Jan–March.",
    },
    "🏗 Engineering Management & MBA": {
        "description": "Bridges technical expertise and business leadership. Ideal for engineers who want to move into management or entrepreneurship.",
        "majors": ["MS Engineering Management", "MBA (Tech Focus)", "MS Management of Technology", "MS Operations Research", "MS Supply Chain Management", "MS Project Management"],
        "careers": ["Product Manager", "Engineering Manager", "Operations Manager", "Management Consultant", "Entrepreneur", "Supply Chain Director"],
        "top_schools": ["MIT Sloan", "Stanford GSB", "Carnegie Mellon Tepper", "Northwestern Kellogg", "Duke Fuqua", "Cornell Tech"],
        "avg_gre": "308–325",
        "avg_cgpa": "7.5–9.0 / 10",
        "application_tips": [
            "Work experience matters more here than in pure technical programs — highlight leadership roles",
            "GMAT is often preferred over GRE for MBA programs — check each school's preference",
            "Essays are more important in business programs — spend significant time on them",
            "Show clear career goals — admissions committees want to see a concrete vision",
            "Recommendation letters from managers or industry supervisors outweigh academic ones here",
        ],
        "timeline": "Round 1: Sept–Oct. Round 2 (most competitive): Jan. Round 3: March–April.",
    },
}

# ── Tabs for each field ──
tab_labels = list(PROGRAM_DATA.keys())
tabs = st.tabs(tab_labels)

for tab, label in zip(tabs, tab_labels):
    prog = PROGRAM_DATA[label]
    with tab:
        # ── Description banner ──
        st.markdown(f'<div class="grad-card"><p style="color:#a8b8d8;font-size:0.95rem;margin:0">{prog["description"]}</p></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1.1, 1], gap="large")

        with col_a:
            # Majors
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">🎓 Available Majors</p>', unsafe_allow_html=True)
            majors_html = "".join(
                f'<span style="display:inline-block;background:#1e2230;border:1px solid #2e3248;'
                f'border-radius:20px;padding:0.25rem 0.75rem;margin:0.2rem;font-size:0.82rem;color:#e8e4dc">{m}</span>'
                for m in prog["majors"]
            )
            st.markdown(f'<div style="margin-bottom:1rem">{majors_html}</div>', unsafe_allow_html=True)

            # Careers
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">💼 Career Paths</p>', unsafe_allow_html=True)
            careers_html = "".join(
                f'<span style="display:inline-block;background:#1a2218;border:1px solid #2e4830;'
                f'border-radius:20px;padding:0.25rem 0.75rem;margin:0.2rem;font-size:0.82rem;color:#7ecb8f">{c}</span>'
                for c in prog["careers"]
            )
            st.markdown(f'<div style="margin-bottom:1rem">{careers_html}</div>', unsafe_allow_html=True)

            # Profile requirements
            st.markdown(f"""
            <div class="grad-card" style="margin-top:0.5rem">
                <p style="font-weight:600;color:#c9a84c;margin-bottom:0.6rem">📋 Typical Profile Requirements</p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.35rem 0">📝 &nbsp;<b style="color:#6b7080">GRE Score</b> &nbsp; <b style="color:#e8e4dc">{prog["avg_gre"]}</b></p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.35rem 0">🎓 &nbsp;<b style="color:#6b7080">CGPA</b> &nbsp; <b style="color:#e8e4dc">{prog["avg_cgpa"]}</b></p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.35rem 0">📅 &nbsp;<b style="color:#6b7080">Timeline</b> &nbsp; <b style="color:#e8e4dc">{prog["timeline"]}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            # Top schools
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.5rem">🏛 Top Schools</p>', unsafe_allow_html=True)
            schools_html = "".join(
                f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0.6rem;margin-bottom:0.3rem;'
                f'background:#181c26;border:1px solid #252834;border-radius:8px">'
                f'<span style="color:#c9a84c;font-size:0.9rem">🏛</span>'
                f'<span style="font-size:0.88rem;color:#e8e4dc">{s}</span></div>'
                for s in prog["top_schools"]
            )
            st.markdown(f'<div style="margin-bottom:1rem">{schools_html}</div>', unsafe_allow_html=True)

            # Application tips — all inside one card
            tips_rows = "".join(
                f'<div style="display:flex;gap:0.6rem;align-items:flex-start;padding:0.5rem 0;'
                f'border-bottom:1px solid #2a2d38">'
                f'<span style="color:#c9a84c;font-weight:700;font-size:0.9rem;min-width:1.2rem">{i}.</span>'
                f'<span style="font-size:0.84rem;color:#c8c4bc;line-height:1.5">{tip}</span>'
                f'</div>'
                for i, tip in enumerate(prog["application_tips"], 1)
            )
            st.markdown(f"""
            <div class="grad-card-accent">
                <p style="font-weight:600;color:#c9a84c;margin-bottom:0.5rem">💡 How to Apply — Key Tips</p>
                {tips_rows}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#3a3e50;font-size:0.78rem">
    GradPath AI · Predictions are probabilistic, not deterministic · Not affiliated with any university
</div>
""", unsafe_allow_html=True)