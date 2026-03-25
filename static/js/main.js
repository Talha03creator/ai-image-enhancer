/*
   VisionAI - Interactive Engine
   Handles SaaS effects, animations, and enhancement workflow
*/

// --- STATE MANAGEMENT ---
let state = {
    uploadedFile: null,
    isProcessing: false,
    originalUrl: null,
    enhancedUrl: null,
    isDemo: false,
    history: JSON.parse(localStorage.getItem('visionAiHistory') || '[]')
};

// --- SELECT ELEMENTS ---
const elements = {
    dropArea: document.getElementById('dropArea'),
    imageInput: document.getElementById('imageInput'),
    previewLayer: document.getElementById('previewLayer'),
    originalImg: document.getElementById('originalImg'),
    enhancedImg: document.getElementById('enhancedImg'),
    enhancedWrapper: document.getElementById('enhancedWrapper'),
    sliderHandle: document.getElementById('sliderHandle'),
    processingSteps: document.getElementById('processingSteps'),
    actionBtns: document.getElementById('actionBtns'),
    enhanceBtn: document.getElementById('enhanceBtn'),
    downloadGroup: document.getElementById('downloadGroup'),
    resetBtn: document.getElementById('resetBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    previewBtn: document.getElementById('previewBtn'),
    previewModal: document.getElementById('previewModal'),
    modalImg: document.getElementById('modalImg'),
    modalContent: document.getElementById('modalContent'),
    closeModal: document.getElementById('closeModal'),
    toastContainer: document.getElementById('toastContainer'),
    counters: document.querySelectorAll('.counter'),
    revealItems: document.querySelectorAll('[data-reveal]'),
    historyGrid: document.getElementById('historyGrid'),
    featureCards: document.querySelectorAll('.feature-card'),
    themeToggle: document.getElementById('themeToggle'),
    heroUploadBtn: document.getElementById('heroUploadBtn'),
    intro: document.getElementById('intro'),
    introLines: document.getElementById('introLines'),
    soundToggle: document.getElementById('soundToggle'),
    voiceToggle: document.getElementById('voiceToggle'),
    modeSelect: document.getElementById('modeSelect'),
    // New Premium elements
    demoBtn: document.getElementById('demoBtn'),
    heroDemoBtn: document.getElementById('heroDemoBtn'),
    aiRecommendation: document.getElementById('aiRecommendation'),
    shareBtn: document.getElementById('shareBtn'),
    compareModesBtn: document.getElementById('compareModesBtn'),
    successModal: document.getElementById('successModal'),
    closeSuccessModal: document.getElementById('closeSuccessModal'),
    successDownloadBtn: document.getElementById('successDownloadBtn'),
    successViewBtn: document.getElementById('successViewBtn'),
    modePreviewHint: document.getElementById('modePreviewHint')
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initCinematicIntro();
    initTheme();
    initAudioPrefs();
    initCursorGlow();
    initParticles();
    initRipples();
    initScrollReveal();
    initCounters();
    initDragAndDrop();
    renderHistory();
    initTiltEffect();
    initModalLogic();
    initHeroCTA();
    // New Premium Init
    initPremiumSaaS();
    initEliteShimmers();
});

// --- CINEMATIC INTRO (BOOT SEQUENCE) ---
function initCinematicIntro() {
    const intro = elements.intro;
    const box = elements.introLines;
    if (!intro || !box) return;

    const hasSeen = sessionStorage.getItem('introSeen') === '1';
    if (hasSeen) {
        intro.style.display = 'none';
        return;
    }
    sessionStorage.setItem('introSeen', '1');

    const lines = [
        "Initializing AI Engine...",
        "Loading Neural Networks...",
        "Preparing Image Enhancement System..."
    ];

    const caret = document.createElement('span');
    caret.className = 'caret';

    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    async function typeLine(text) {
        const lineEl = document.createElement('div');
        lineEl.className = 'intro-line';
        box.appendChild(lineEl);
        lineEl.textContent = '';
        lineEl.appendChild(caret);

        for (let i = 0; i < text.length; i++) {
            lineEl.textContent = text.slice(0, i + 1);
            lineEl.appendChild(caret);
            await sleep(18 + Math.random() * 18);
        }
        await sleep(240);
        // freeze final line without caret
        caret.remove();
        lineEl.textContent = text;
        await sleep(240);
    }

    async function run() {
        // initial black screen moment
        await sleep(250);
        for (const l of lines) await typeLine(l);
        await sleep(800);
        intro.classList.add('hidden');
        await sleep(950);
        intro.style.display = 'none';
    }

    const skip = () => {
        intro.classList.add('hidden');
        setTimeout(() => {
            intro.style.display = 'none';
        }, 950);
    };

    intro.addEventListener('click', skip, { once: true });
    run();
}

// --- THEME TOGGLE (DARK/LIGHT) ---
function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') document.body.classList.add('dark');
    if (saved === 'light') document.body.classList.remove('dark');

    if (!elements.themeToggle) return;
    elements.themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light');
    });
}

// --- AUDIO + VOICE PREFERENCES (USER-GESTURE SAFE) ---
const ui = {
    audioReady: false,
    audioEnabled: localStorage.getItem('sound') ? localStorage.getItem('sound') === '1' : true,
    voiceEnabled: localStorage.getItem('voice') ? localStorage.getItem('voice') === '1' : false,
    audioCtx: null
};

function initAudioPrefs() {
    // reflect saved state in UI
    if (elements.soundToggle) elements.soundToggle.classList.toggle('off', !ui.audioEnabled);
    if (elements.voiceToggle) elements.voiceToggle.classList.toggle('off', !ui.voiceEnabled);

    elements.soundToggle?.addEventListener('click', () => {
        ui.audioEnabled = !ui.audioEnabled;
        localStorage.setItem('sound', ui.audioEnabled ? '1' : '0');
        elements.soundToggle.classList.toggle('off', !ui.audioEnabled);
        if (ui.audioEnabled) beep('toggle');
    });

    elements.voiceToggle?.addEventListener('click', () => {
        ui.voiceEnabled = !ui.voiceEnabled;
        localStorage.setItem('voice', ui.voiceEnabled ? '1' : '0');
        elements.voiceToggle.classList.toggle('off', !ui.voiceEnabled);
        if (ui.voiceEnabled) speak("AI voice enabled.");
    });

    // Create/resume AudioContext only after a user gesture
    const arm = async () => {
        if (ui.audioReady) return;
        try {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            ui.audioCtx = new Ctx();
            if (ui.audioCtx.state === 'suspended') await ui.audioCtx.resume();
            ui.audioReady = true;
        } catch {
            // ignore: audio unavailable
        }
    };
    window.addEventListener('pointerdown', arm, { once: true, passive: true });
    window.addEventListener('keydown', arm, { once: true });
}

function beep(kind = 'click') {
    if (!ui.audioEnabled || !ui.audioReady || !ui.audioCtx) return;
    const ctx = ui.audioCtx;
    const t0 = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    const map = {
        click: { f: 520, a: 0.05, d: 0.06, type: 'triangle' },
        hover: { f: 740, a: 0.02, d: 0.04, type: 'sine' },
        success: { f: 660, a: 0.05, d: 0.14, type: 'sine' },
        toggle: { f: 880, a: 0.03, d: 0.07, type: 'triangle' }
    };
    const p = map[kind] || map.click;

    osc.type = p.type;
    osc.frequency.setValueAtTime(p.f, t0);
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(p.a, t0 + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + p.d);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + p.d + 0.02);
}

function speak(text) {
    if (!ui.voiceEnabled) return;
    if (!('speechSynthesis' in window)) return;
    try {
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance(text);
        msg.rate = 1;
        msg.pitch = 1.08;
        msg.volume = 0.95;
        window.speechSynthesis.speak(msg);
    } catch {
        // ignore
    }
}

// --- HERO CTA (UPLOAD BUTTON) ---
function initHeroCTA() {
    if (!elements.heroUploadBtn) return;
    elements.heroUploadBtn.addEventListener('click', () => {
        // Keep UX snappy: scroll to enhancer, then open the same file input used by drag/drop.
        const target = document.getElementById('enhancer');
        if (target) target.scrollIntoView({ behavior: 'smooth' });
        // Small delay to allow scroll / layout settle (especially on mobile)
        setTimeout(() => {
            if (elements.imageInput) elements.imageInput.click();
        }, 250);
    });
}

// --- CURSOR GLOW ---
function initCursorGlow() {
    const glow = document.querySelector('.cursor-glow');
    if (!glow) return;

    // Disable on touch devices
    const isCoarse = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
    if (isCoarse) {
        glow.style.display = 'none';
        return;
    }

    let x = -9999, y = -9999;
    let raf = null;
    const render = () => {
        glow.style.left = `${x}px`;
        glow.style.top = `${y}px`;
        raf = null;
    };

    const onMove = (e) => {
        x = e.clientX;
        y = e.clientY;
        glow.style.opacity = '1';
        if (!raf) raf = requestAnimationFrame(render);
    };

    const onLeave = () => {
        glow.style.opacity = '0';
    };

    document.addEventListener('mousemove', onMove, { passive: true });
    window.addEventListener('blur', onLeave);
    document.addEventListener('mouseleave', onLeave);

    // Expand on click for “interactive cursor”
    document.addEventListener('mousedown', () => {
        glow.style.transform = 'translate(-50%, -50%) scale(1.35)';
    });
    document.addEventListener('mouseup', () => {
        glow.style.transform = 'translate(-50%, -50%) scale(1)';
    });
}

// --- CANVAS PARTICLES (HERO) ---
function initParticles() {
    const canvas = document.getElementById('particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0, h = 0, dpr = 1;
    const particles = [];
    const COUNT = 70;

    function resize() {
        dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        w = canvas.clientWidth || window.innerWidth;
        h = canvas.clientHeight || window.innerHeight;
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function seed() {
        particles.length = 0;
        for (let i = 0; i < COUNT; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                r: Math.random() * 2.2 + 0.4,
                dx: (Math.random() - 0.5) * 0.45,
                dy: (Math.random() - 0.5) * 0.45,
                a: Math.random() * 0.55 + 0.15
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);

        // Particles
        for (const p of particles) {
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < -20) p.x = w + 20;
            if (p.x > w + 20) p.x = -20;
            if (p.y < -20) p.y = h + 20;
            if (p.y > h + 20) p.y = -20;

            ctx.beginPath();
            ctx.fillStyle = `rgba(0, 255, 255, ${p.a})`;
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        }

        // Light connections (cheap + premium)
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const a = particles[i], b = particles[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 140) {
                    const alpha = (1 - dist / 140) * 0.08;
                    ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(draw);
    }

    resize();
    seed();
    draw();
    window.addEventListener('resize', () => {
        resize();
        seed();
    }, { passive: true });
}

// --- BUTTON RIPPLE MICRO-INTERACTIONS ---
function initRipples() {
    const targets = document.querySelectorAll('.btn-primary, .btn-primary-sm, .btn-secondary, .btn-enhance, .btn-download, .btn-preview, .btn-reset');
    targets.forEach(btn => {
        btn.addEventListener('pointerenter', () => beep('hover'), { passive: true });
        btn.addEventListener('click', (e) => {
            beep('click');
            const rect = btn.getBoundingClientRect();
            const ripple = document.createElement('span');
            ripple.className = 'ripple';
            ripple.style.left = `${e.clientX - rect.left}px`;
            ripple.style.top = `${e.clientY - rect.top}px`;
            btn.appendChild(ripple);
            setTimeout(() => ripple.remove(), 650);
        }, { passive: true });
    });
}

// --- TOAST NOTIFICATIONS ---
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = type === 'success' ? 'check-circle' : 'alert-circle';
    toast.innerHTML = `
        <i data-lucide="${icon}"></i>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    lucide.createIcons();
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'toastSlideOut 0.4s forwards';
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}

// --- 3D TILT EFFECT ---
function initTiltEffect() {
    const cards = [...elements.featureCards, ...(elements.historyGrid ? elements.historyGrid.querySelectorAll('.history-item') : [])];
    
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const xc = rect.width / 2;
            const yc = rect.height / 2;
            
            const dx = x - xc;
            const dy = y - yc;
            
            card.style.transform = `rotateY(${dx / 10}deg) rotateX(${-dy / 10}deg) translateZ(10px)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'rotateY(0) rotateX(0)';
        });
    });
}

// --- MODAL LOGIC ---
function initModalLogic() {
    elements.previewBtn.addEventListener('click', () => {
        if (!state.enhancedUrl) return;
        elements.modalImg.src = state.enhancedUrl;
        elements.previewModal.classList.add('active');
    });
    
    elements.closeModal.addEventListener('click', () => {
        elements.previewModal.classList.remove('active');
        elements.modalContent.classList.remove('zoomed');
    });
    
    elements.previewModal.addEventListener('click', (e) => {
        if (e.target === elements.previewModal) {
            elements.previewModal.classList.remove('active');
            elements.modalContent.classList.remove('zoomed');
        }
    });
    
    elements.modalContent.addEventListener('click', () => {
        elements.modalContent.classList.toggle('zoomed');
    });
}

// --- SCROLL REVEAL ---
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.1 });

    elements.revealItems.forEach(item => observer.observe(item));
}

// --- STATS COUNTER ---
function initCounters() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                startCounter(entry.target);
            }
        });
    }, { threshold: 1.0 });

    elements.counters.forEach(counter => observer.observe(counter));
}

function startCounter(el) {
    const target = parseInt(el.getAttribute('data-target'));
    let count = 0;
    const increment = target / 50;
    const interval = setInterval(() => {
        count += increment;
        if (count >= target) {
            el.innerText = target + (target === 1000 ? '+' : target === 99 ? '%' : '%');
            clearInterval(interval);
        } else {
            el.innerText = Math.floor(count);
        }
    }, 30);
}

// --- DRAG AND DROP ---
function initDragAndDrop() {
    elements.dropArea.addEventListener('click', (e) => {
        // Prevent click if an image is already uploaded unless clicking reset or explicitly adding new
        if (state.uploadedFile) {
            // Only allow if clicking specific placeholder elements (if they were visible)
            if (!e.target.closest('.upload-placeholder')) {
                e.preventDefault();
                return;
            }
        }
        elements.imageInput.click();
    });
    
    elements.imageInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    ['dragenter', 'dragover'].forEach(name => {
        elements.dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            elements.dropArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        elements.dropArea.addEventListener(name, (e) => {
            e.preventDefault();
            elements.dropArea.classList.remove('dragover');
        });
    });

    elements.dropArea.addEventListener('drop', (e) => {
        handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file) return;

    // Robust validation: Check MIME type OR file extension fallback
    const validExtensions = ['jpg', 'jpeg', 'png', 'webp'];
    const fileExt = file.name ? file.name.split('.').pop().toLowerCase() : '';
    const isImage = file.type.startsWith('image/') || validExtensions.includes(fileExt);

    if (!isImage) {
        showError(`Invalid file: "${file.name}". Please upload a JPG, PNG, or WebP image.`);
        return;
    }

    // File size validation (match backend: 30MB max)
    const MAX_SIZE = 30 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
        showError(`File too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB. Max limit is 30MB.`);
        return;
    }

    state.uploadedFile = file;
    state.originalUrl = URL.createObjectURL(file);
    speak("Image received. Starting enhancement.");
    
    // Reset enhanced URL state
    state.enhancedUrl = null;
    if (elements.aiRecommendation) elements.aiRecommendation.classList.add('hidden');

    // Premium Feature: Image Type Detection (Frontend Only)
    detectImageProperties(file);

    // Attach listener BEFORE setting src to ensure it catches the load event
    elements.originalImg.onload = () => {
        elements.previewLayer.classList.remove('hidden');
        document.querySelector('.upload-placeholder').classList.add('hidden');
        
        // Ensure buttons are visible
        elements.enhanceBtn.classList.remove('hidden');
        elements.enhanceBtn.querySelector('span').innerText = "Enhance Image";
        elements.downloadGroup.classList.add('hidden');

        // Use requestAnimationFrame to ensure the DOM has updated dimensions
        requestAnimationFrame(() => {
            const containerWidth = elements.previewLayer.offsetWidth;
            elements.enhancedWrapper.style.setProperty('--container-width', containerWidth + 'px');
            updateSlider(50);
        });
    };

    // Now set the sources
    elements.originalImg.src = state.originalUrl;
    elements.enhancedImg.src = state.originalUrl; // Placeholder
}

// --- BEFORE/AFTER SLIDER ---
let isDragging = false;

function updateSlider(percentage) {
    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;
    
    // Set width and CSS variable for the clipping logic
    elements.enhancedWrapper.style.width = `${percentage}%`;
    elements.sliderHandle.style.left = `${percentage}%`;
    
    // This allows CSS to fix the image width dynamically
    elements.enhancedWrapper.style.setProperty('--slider-pos', percentage);
    
    // Ensure the inner image stays the full width of the container
    const containerWidth = elements.previewLayer.offsetWidth;
    if (containerWidth > 0) {
        elements.enhancedImg.style.width = containerWidth + "px";
        elements.enhancedWrapper.style.setProperty('--container-width', containerWidth + 'px');
    }
}

elements.sliderHandle.addEventListener('mousedown', () => isDragging = true);
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const rect = elements.previewLayer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    updateSlider(percentage);
});

// Touch support for slider
elements.sliderHandle.addEventListener('touchstart', () => isDragging = true);
window.addEventListener('touchend', () => isDragging = false);
window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const touch = e.touches[0];
    const rect = elements.previewLayer.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    updateSlider(percentage);
});

// --- AI ENHANCEMENT WORKFLOW ---
elements.enhanceBtn.addEventListener('click', async () => {
    if (!state.uploadedFile || state.isProcessing) return;
    
    startProcessingUI();
    
    const formData = new FormData();
    formData.append("file", state.uploadedFile);
    const selectedMode = elements.modeSelect ? elements.modeSelect.value : "auto";
    formData.append("filter", selectedMode || "auto"); // Default to auto-AI

    try {
        // Run multi-step animation sequence (Real logic starts after)
        runStepAnimation();

        // Use same-origin base when possible
        const API_BASE = window.location?.origin || "http://127.0.0.1:8000";

        // Skip polling /status in the new direct flow as requested
        console.log("Starting direct enhancement call...");

        // 2. Perform Enhancement
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 305000); // 5 min + buffer

        const slowWarningId = setTimeout(() => {
            if (state.isProcessing) {
                const finalStep = document.querySelector('.step[data-step="4"] span');
                if (finalStep) finalStep.innerText = "Processing on CPU (This may take 1-2 minutes)...";
            }
        }, 30000);

        const response = await fetch(`${API_BASE}/enhance`, {
            method: "POST",
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        clearTimeout(slowWarningId);

        // Backend returns JSON: { status: "success", image_url: "/static/outputs/..." }
        const data = await response.json().catch(() => null);
        if (!response.ok) {
            const detail = data?.detail || data?.message || "Enhancement failed";
            throw new Error(detail);
        }
        if (!data || data.status !== "success" || !data.image_url) {
            throw new Error(data?.message || "Unexpected server response.");
        }

        // Resolve relative URLs against the server origin
        const resolvedUrl = new URL(data.image_url, API_BASE).toString();
        state.enhancedUrl = resolvedUrl;
        elements.enhancedImg.src = resolvedUrl;

        addToHistory(resolvedUrl);
        
        requestAnimationFrame(() => {
            const containerWidth = elements.previewLayer.offsetWidth;
            elements.enhancedWrapper.style.setProperty('--container-width', containerWidth + 'px');
            finishProcessingUI(true);
            showToast("✅ Your image has been enhanced successfully! You can now preview or download it.");
            // Cinematic success moment
            document.body.classList.add('flash-success');
            setTimeout(() => document.body.classList.remove('flash-success'), 1100);
            beep('success');
            speak("Enhancement complete. Your image is ready.");
        });
    } catch (err) {
        console.error(err);
        let msg = err.message;
        if (err.name === 'AbortError') msg = "Request timed out. The image was too complex or the server is overloaded.";
        finishProcessingUI(false, msg);
        showToast(msg, 'error');
    }
});

async function runStepAnimation() {
    const stepEls = document.querySelectorAll('.step');
    for (let i = 0; i < stepEls.length; i++) {
        stepEls.forEach(s => s.classList.remove('active'));
        stepEls[i].classList.add('active');
        await new Promise(r => setTimeout(r, 600)); 
    }
}

function startProcessingUI() {
    state.isProcessing = true;
    elements.enhanceBtn.disabled = true;
    elements.enhanceBtn.classList.add('hidden');
    
    // Reset status text in case of retry
    const finalStep = document.querySelector('.step[data-step="4"] span');
    if (finalStep) finalStep.innerText = "Finalizing...";
    
    // Re-render processing steps to clear previous errors
    elements.processingSteps.innerHTML = `
        <div class="ai-orb" aria-hidden="true"></div>
        <div class="step" data-step="1">
            <i data-lucide="scan"></i>
            <span>Analyzing image texture...</span>
        </div>
        <div class="step" data-step="2">
            <i data-lucide="layers"></i>
            <span>Enhancing deep details...</span>
        </div>
        <div class="step" data-step="3">
            <i data-lucide="sparkles"></i>
            <span>Optimizing lighting...</span>
        </div>
        <div class="step" data-step="4">
            <i data-lucide="check-circle"></i>
            <span id="aiStatusText">Finalizing...</span>
        </div>
    `;
    lucide.createIcons();
    elements.processingSteps.classList.remove('hidden');
    elements.downloadGroup.classList.add('hidden');

    // Premium “AI thoughts” micro-copy loop (UI only)
    clearInterval(window.__aiStatusInterval);
    const phrases = [
        "Analyzing pixels…",
        "Enhancing details…",
        "Reconstructing clarity…",
        "Refining edges…",
        "Finalizing output…"
    ];
    let idx = 0;
    window.__aiStatusInterval = setInterval(() => {
        if (!state.isProcessing) return;
        const el = document.getElementById('aiStatusText');
        if (!el) return;
        idx = (idx + 1) % phrases.length;
        el.textContent = phrases[idx];
    }, 1400);
}

function finishProcessingUI(success, errorMsg = "") {
    state.isProcessing = false;
    clearInterval(window.__aiStatusInterval);
    elements.enhanceBtn.disabled = false;
    
    if (success) {
        elements.processingSteps.classList.add('hidden');
        elements.downloadGroup.classList.remove('hidden');
        
        // Premium Reveal Animation
        triggerRevealSequence();
    } else {
        elements.processingSteps.innerHTML = `
            <div class="step-error">
                <i data-lucide="alert-circle"></i>
                <span>${errorMsg || "Enhancement failed."}</span>
            </div>
        `;
        lucide.createIcons();
        elements.enhanceBtn.classList.remove('hidden');
        elements.enhanceBtn.querySelector('span').innerText = "Try Again";
    }
}

// --- UTILS ---
function showError(msg) {
    alert(msg); // In a production app, we'd use a custom toast
}

// --- HISTORY MANAGEMENT ---
function addToHistory(url) {
    // Note: In a real app, we'd store the image in IndexedDB or a server.
    // For this SaaS demo, we'll store the object URL temporarily 
    // and just keep the session history.
    state.history.unshift({
        url: url,
        timestamp: new Date().toLocaleTimeString()
    });
    
    // Limit history to 10 items
    if (state.history.length > 10) state.history.pop();
    
    localStorage.setItem('visionAiHistory', JSON.stringify(state.history));
    renderHistory();
}

function renderHistory() {
    if (!elements.historyGrid) return;
    
    if (state.history.length === 0) {
        elements.historyGrid.innerHTML = '<div class="no-history">No images enhanced yet. Get started above!</div>';
        return;
    }
    
    elements.historyGrid.innerHTML = state.history.map((item, index) => `
        <div class="history-item" data-index="${index}" onclick="viewFromHistory(${index})">
            <img src="${item.url}" alt="Enhanced History">
            <div class="history-overlay">
                <span><i data-lucide="clock" style="width:12px"></i> ${item.timestamp}</span>
                <i data-lucide="external-link" style="width:12px"></i>
            </div>
        </div>
    `).join('');
    
    if (window.lucide) lucide.createIcons();
}

window.viewFromHistory = function(index) {
    const item = state.history[index];
    if (item) {
        // Simple preview logic
        const link = document.createElement('a');
        link.href = item.url;
        link.target = "_blank";
        link.click();
    }
};

elements.resetBtn.addEventListener('click', () => {
    state.uploadedFile = null;
    elements.previewLayer.classList.add('hidden');
    document.querySelector('.upload-placeholder').classList.remove('hidden');
    elements.enhanceBtn.classList.remove('hidden');
    elements.enhanceBtn.querySelector('span').innerText = "Enhance Image";
    elements.downloadGroup.classList.add('hidden');
    elements.imageInput.value = ''; // Reset input
});

elements.downloadBtn.addEventListener('click', (e) => {
    if (!state.enhancedUrl) {
        e.preventDefault();
        return;
    }
});

// --- PREMIUM SAAS UPGRADE LOGIC ---

function initPremiumSaaS() {
    // 1. Demo Mode Listeners
    if (elements.demoBtn) elements.demoBtn.addEventListener('click', loadDemoImage);
    if (elements.heroDemoBtn) elements.heroDemoBtn.addEventListener('click', loadDemoImage);

    // 2. Success Modal Listeners
    if (elements.closeSuccessModal) {
        elements.closeSuccessModal.addEventListener('click', () => {
            elements.successModal.classList.remove('active');
        });
    }
    if (elements.successDownloadBtn) {
        elements.successDownloadBtn.addEventListener('click', () => elements.downloadBtn.click());
    }
    if (elements.successViewBtn) {
        elements.successViewBtn.addEventListener('click', () => {
            elements.successModal.classList.remove('active');
            elements.previewBtn.click();
        });
    }

    // 3. Social Share
    if (elements.shareBtn) {
        elements.shareBtn.addEventListener('click', async () => {
            if (!state.enhancedUrl) return;
            if (navigator.share) {
                try {
                    await navigator.share({
                        title: 'VisionAI Enhanced Image',
                        text: 'Look at this amazing AI enhancement!',
                        url: state.enhancedUrl
                    });
                } catch (err) { /* ignore cancel */ }
            } else {
                navigator.clipboard.writeText(state.enhancedUrl);
                showToast("Link copied to clipboard! 🔗");
            }
        });
    }

    // 4. Mode Preview on Hover
    // Since <option> hover is limited, we show a hint and apply classes via JS
    elements.modeSelect?.addEventListener('change', () => {
        const val = elements.modeSelect.value;
        // Apply temporary filter to original image as a "preview"
        elements.originalImg.className = `preview-${val}`;
        setTimeout(() => elements.originalImg.className = '', 1500);
    });

    // 5. Compare Modes side-by-side
    elements.compareModesBtn?.addEventListener('click', () => {
        elements.previewLayer.classList.toggle('compare-active');
        const active = elements.previewLayer.classList.contains('compare-active');
        elements.compareModesBtn.innerHTML = active ? 
            '<i data-lucide="x-circle"></i> Exit Comparison' : 
            '<i data-lucide="split-square-horizontal"></i> Compare Mode Views';
        lucide.createIcons();
    });
}

function loadDemoImage() {
    // Scroll to enhancer
    const target = document.getElementById('enhancer');
    if (target) target.scrollIntoView({ behavior: 'smooth' });
    
    state.isDemo = true;
    showToast("✨ Loading high-res sample image...");
    
    // Using demo_sample.jpg which exists in the static directory
    // We'll fetch it and convert to a file object to simulate upload
    fetch('/static/demo_sample.jpg')
        .then(res => res.blob())
        .then(blob => {
            const file = new File([blob], "demo_sample.jpg", { type: "image/jpeg" });
            setTimeout(() => handleFile(file), 500);
        });
}

function detectImageProperties(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 100; // Small sample
            brightness /= (data.length / 4);
            
            // Elite: Show Scanning Line
            const scanner = document.getElementById('scanningLine');
            if (scanner) scanner.classList.remove('hidden');

            setTimeout(() => {
                if (scanner) scanner.classList.add('hidden');
                
                // Logic for Recommendation
                let typeMsg = "";
                let recoMode = "ultra_hd";
                let recoLabel = "Ultra HD 🔥";

                if (brightness < 80) {
                    typeMsg = "Detected: Low Light Image 🌙";
                    recoMode = "low_light";
                    recoLabel = "Low-Light Fix 🌙";
                } else if (img.width < 1200) {
                    typeMsg = "Detected: Low Res Image 📷";
                    recoMode = "ultra_hd";
                    recoLabel = "Ultra HD Boost 🔥";
                } else {
                    typeMsg = "Detected: Standard Landscape 🏞️";
                    recoMode = "hdr";
                    recoLabel = "HDR Boost 🌅";
                }

                if (elements.aiRecommendation) {
                    elements.aiRecommendation.classList.remove('hidden');
                    // Elite: Typing Effect
                    typeWriter(elements.aiRecommendation, `<span>${typeMsg} &nbsp;|&nbsp; <b>Recommended Mode: ${recoLabel}</b></span>`);
                    
                    // Auto-set the mode
                    if (elements.modeSelect) {
                        elements.modeSelect.value = recoMode;
                    }
                }
            }, 2000); // 2 seconds of scanning
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function triggerRevealSequence() {
    // 1. Blur-to-Sharp Transition
    elements.enhancedImg.classList.add('reveal-sharp');
    
    // 2. Slider Sweep
    updateSlider(0);
    setTimeout(() => {
        updateSlider(100);
        setTimeout(() => {
            updateSlider(50);
            
            // 3. Show Success WOW Modal
            setTimeout(() => {
                if (elements.successModal) {
                    elements.successModal.classList.add('active');
                    fireConfetti();
                    beep('success');
                }
            }, 800);
        }, 800);
    }, 400);

    setTimeout(() => {
        elements.enhancedImg.classList.remove('reveal-sharp');
    }, 2000);
}

// --- ELITE UI HELPERS ---

function initEliteShimmers() {
    const pBtns = [elements.enhanceBtn, elements.downloadBtn, elements.heroUploadBtn];
    pBtns.forEach(btn => {
        if (btn) btn.classList.add('btn-shimmer');
    });
}

function typeWriter(element, html) {
    element.innerHTML = '<i data-lucide="brain-circuit" style="width:18px"></i> <span class="typing-text"></span>';
    const textSpan = element.querySelector('.typing-text');
    lucide.createIcons();
    
    // Quick trick to handle HTML in typing: 
    // We'll just type the plain text but set innerHTML at the end for formatting
    const temp = document.createElement('div');
    temp.innerHTML = html;
    const fullText = temp.textContent;
    
    let i = 0;
    const interval = setInterval(() => {
        textSpan.textContent += fullText.charAt(i);
        i++;
        if (i >= fullText.length) {
            clearInterval(interval);
            element.innerHTML = `<i data-lucide="brain-circuit" style="width:18px"></i> ${html}`;
            lucide.createIcons();
        }
    }, 30);
}

function fireConfetti() {
    const container = document.getElementById('confettiContainer');
    if (!container) return;
    container.innerHTML = '';
    
    const colors = ['#3b82f6', '#8b5cf6', '#6366f1', '#a855f7', '#00ffff'];
    
    for (let i = 0; i < 50; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + '%';
        confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.animationDelay = Math.random() * 0.5 + 's';
        confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
        container.appendChild(confetti);
        
        // Cleanup
        setTimeout(() => confetti.remove(), 3000);
    }
}

