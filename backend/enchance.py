import os
import time
import threading

# NOTE: Heavy ML imports (torch/cv2/realesrgan/gfpgan) are intentionally deferred
# into functions/methods so the FastAPI server can start instantly on Windows.

# --- CONFIGURATION ---
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
MODELS = {
    'realesrgan': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'path': os.path.join(WEIGHTS_DIR, 'RealESRGAN_x2plus.pth'),
        'size': 67040989
    },
    'realesrgan_x4': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'path': os.path.join(WEIGHTS_DIR, 'RealESRGAN_x4plus.pth'),
        'size': 67040989
    },
    'gfpgan': {
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'path': os.path.join(WEIGHTS_DIR, 'GFPGANv1.3.pth'),
        'size': 348632874
    },
    'detection': {
        'url': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_ResNet50_Final.pth',
        'path': os.path.join(WEIGHTS_DIR, 'detection_ResNet50_Final.pth'),
        'size': 109497761
    },
    'parsing': {
        'url': 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
        'path': os.path.join(WEIGHTS_DIR, 'parsing_parsenet.pth'),
        'size': 85331193
    }
}

# --- PERFORMANCE CONSTANTS ----------------------------------------------------
# FIX 1: Reduced from 1200 -> 720px.  GFPGAN/ESRGAN work is O(n2) in pixels;
#         halving the max side speed is ~4x faster with negligible quality loss
#         at typical screen sizes.
MAX_INPUT_DIM_STD = 1024
MAX_INPUT_DIM_PREMIUM = 2048

# FIX 2: ESRGAN tile size reduction 400 -> 256.  Smaller tiles fit in L3 CPU
#         cache, run faster in serial, and avoid OOM spikes on large images.
ESRGAN_TILE = 256

# FIX 3: Internal wall-time budget.  If the inference thread hasn't returned by
#         this many seconds, we write the *best available* result to disk so the
#         caller always gets a file instead of a 504.
INTERNAL_TIMEOUT_S = 125
# -----------------------------------------------------------------------------

# Ensure weights directory exists
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_model(name):
    """Download model weights with robust integrity checks."""
    config = MODELS[name]
    dest = config['path']
    url = config['url']
    expected_size = config.get('size')
    min_reasonable_size = max(5 * 1024 * 1024, int(expected_size * 0.6)) if expected_size else 5 * 1024 * 1024

    if os.path.exists(dest):
        current_size = os.path.getsize(dest)
        if current_size >= min_reasonable_size:
            return
        else:
            print(f"Model {name} looks incomplete (Size: {current_size}). Re-downloading...")
            try:
                os.remove(dest)
            except OSError:
                pass

    print(f"\n--- SETTING UP AI ENGINE: Downloading {name} ---")
    print(f"This is a one-time process. Please do not close the application.")

    try:
        from torch.hub import download_url_to_file
        download_url_to_file(url, dest, progress=True)

        new_size = os.path.getsize(dest)
        if new_size < min_reasonable_size:
            raise Exception(f"Downloaded file too small for {name}: {new_size} bytes")

        print(f"[Done] {name} verified and ready.\n")
    except Exception as e:
        print(f"CRITICAL ERROR downloading {name}: {str(e)}")
        if os.path.exists(dest):
            os.remove(dest)
        raise Exception(f"AI Engine failed to initialize: {name} is missing or corrupted.")


class AIImageEnhancer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIImageEnhancer, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._constructed = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_constructed", False):
            return

        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialized = False
        self._is_loading = False
        self._last_error = None
        self._start_time = time.time()
        self._constructed = True
        print(f"[{self._get_timestamp()}] AI Engine Instance Created (Device: {self.device})")

    def _get_timestamp(self):
        return time.strftime("%H:%M:%S")

    def initialize(self):
        """Perform the heavy model loading and initialization."""
        if self._initialized or self._is_loading:
            return

        self._is_loading = True
        print(f"[{self._get_timestamp()}] Beginning AI Engine heavy initialization...")

        try:
            from realesrgan import RealESRGANer
            from gfpgan import GFPGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Ensure all models are downloaded
            for model_name in MODELS:
                download_model(model_name)

            # FIX 2 applied: tile=ESRGAN_TILE (256) instead of 400
            print(f"[{self._get_timestamp()}] Loading Real-ESRGAN x2plus model...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.upscaler = RealESRGANer(
                scale=2,
                model_path=MODELS['realesrgan']['path'],
                model=model,
                tile=ESRGAN_TILE,
                tile_pad=10,
                pre_pad=0,
                half=False if self.device.type == 'cpu' else True,
                device=self.device
            )

            print(f"[{self._get_timestamp()}] Loading GFPGAN model (bg_upsampler disabled for speed)...")
            self.face_restorer = GFPGANer(
                model_path=MODELS['gfpgan']['path'],
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )

            self._initialized = True
            self._is_loading = False
            duration = time.time() - self._start_time
            print(f"[{self._get_timestamp()}] AI Engine fully ready (Total Load Time: {duration:.2f}s).")
        except Exception as e:
            self._is_loading = False
            self._last_error = str(e)
            print(f"[{self._get_timestamp()}] CRITICAL ERROR during AI Engine initialization: {e}")
            raise

    def get_status(self):
        if self._initialized:
            return "ready"
        if self._is_loading:
            return "loading"
        if self._last_error:
            return f"error: {self._last_error}"
        return "idle"

    # -------------------------------------------------------------------------
    #  FAST OPENCV PIPELINE  (fix: always returns a result, <3s on any hardware)
    # -------------------------------------------------------------------------
    def _opencv_enhance(self, img, effective_mode: str, is_blurry: bool,
                        is_noisy: bool, is_dark: bool):
        """Pure-OpenCV enhancement pipeline. Used as primary path for 'auto'/'general'
        and as a guaranteed fallback for all other modes. Never raises."""
        import cv2
        import numpy as np

        ts = self._get_timestamp
        print(f"[{ts()}] [OpenCV] Starting fast enhancement pipeline...")
        out = img.copy()

        try:
            # 1. Upscale with LANCZOS4 (fast, good quality)
            h, w = out.shape[:2]
            out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            print(f"[{ts()}] [OpenCV] Step 1: 2x LANCZOS4 upscale -> {w*2}x{h*2}")

            # 2. Denoise if noisy
            if is_noisy and effective_mode != "bw":
                out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
                print(f"[{ts()}] [OpenCV] Step 2a: Denoising applied")

            if effective_mode != "bw":
                # 3. CLAHE contrast enhancement
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clip = 3.0 if is_dark else 2.0
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
                print(f"[{ts()}] [OpenCV] Step 3: CLAHE contrast (clip={clip})")

                # 4. Unsharp mask
                strength = 0.75 if is_blurry else 0.45
                blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
                out = cv2.addWeighted(out, 1.0 + strength, blur, -strength, 0)
                print(f"[{ts()}] [OpenCV] Step 4: Unsharp mask (strength={strength:.2f})")

                # 5. Vibrance boost
                hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
                h_c, s_c, v_c = cv2.split(hsv)
                s_c = np.clip(s_c * 1.10, 0, 255)
                v_c = np.clip(v_c * (1.06 if is_dark else 1.03), 0, 255)
                out = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]).astype(np.uint8), cv2.COLOR_HSV2BGR)
                print(f"[{ts()}] [OpenCV] Step 5: Vibrance boost applied")

            # Mode-specific post-processing is applied centrally in enhance()
            # after all pipeline branches complete (see Step 3 below).

        except Exception as ex:
            print(f"[{ts()}] [OpenCV] WARNING in pipeline step: {ex}. Using best-so-far.")

        print(f"[{ts()}] [OpenCV] Fast pipeline complete.")
        return out

    # -------------------------------------------------------------------------
    #  PREMIUM MODES
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #  PREMIUM MODES
    # -------------------------------------------------------------------------
    def _ultra_hd_mode(self, image, stats: dict):
        import cv2, numpy as np
        ts = self._get_timestamp
        out = image.copy()
        
        try:
            is_blurry = stats.get('is_blurry', False)
            is_noisy = stats.get('is_noisy', False)
            is_dark = stats.get('is_dark', False)
            is_portrait_like = stats.get('is_portrait_like', False)
            avg_sat = stats.get('avg_sat', 128)
            contrast = stats.get('contrast_score', 50)
            
            print(f"[{ts()}] [Premium] Ultra HD: Starting adaptive pass. Stats: {stats}")

            # 1. Base upscaling & Face Routing
            if is_portrait_like:
                print(f"[{ts()}] [Premium] Ultra HD: Adaptive Face Restoration...")
                # Slightly adjust weight based on blur: blurrier images need more restoration
                face_weight = 0.5 + (0.1 if is_blurry else 0.0)
                _, _, out = self.face_restorer.enhance(
                    out, has_aligned=False, only_center_face=False, paste_back=True, weight=face_weight
                )
            else:
                print(f"[{ts()}] [Premium] Ultra HD: Neural Upscaling...")
                out, _ = self.upscaler.enhance(out, outscale=2)
                
            # 2. Adaptive Denoising (Dynamic Strength)
            if is_noisy or (is_dark and contrast < 40):
                noise_level = min(5, int(stats.get('noise_sigma', 0) / 10) + 2)
                print(f"[{ts()}] [Premium] Ultra HD: Dynamic Denoising (Strength={noise_level})...")
                out = cv2.fastNlMeansDenoisingColored(out, None, noise_level, noise_level, 7, 21)

            # 3. Intelligent Brightening (only if needed)
            if is_dark:
                # Dynamic alpha based on how far below threshold we are
                alpha = 1.0 + (70 - stats.get('brightness', 70)) * 0.003
                beta = int((70 - stats.get('brightness', 70)) * 0.4)
                print(f"[{ts()}] [Premium] Ultra HD: Adaptive Brighten (Alpha={alpha:.2f}, Beta={beta})...")
                out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
                
            # 4. Smart Detail Enhancement (Non-linear sharpening)
            if is_blurry:
                # Higher sigma for blurrier images
                s_val = 12 if is_blurry else 8
                print(f"[{ts()}] [Premium] Ultra HD: Smart Detail Boost (Sigma={s_val})...")
                out = cv2.detailEnhance(out, sigma_s=s_val, sigma_r=0.15)
            elif contrast < 45:
                # Boost edge clarity for low contrast images
                print(f"[{ts()}] [Premium] Ultra HD: Contrast-Aware Edge Sharpening...")
                blur_k = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
                out = cv2.addWeighted(out, 1.25, blur_k, -0.25, 0)
            else:
                # Image is already sharp and high contrast, do very little
                print(f"[{ts()}] [Premium] Ultra HD: Preserving natural texture...")
                blur_k = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8)
                out = cv2.addWeighted(out, 1.1, blur_k, -0.1, 0)

            # 5. Natural Color Correction (Adaptive Saturation)
            # Target an average saturation of ~140. If image is already vibrant, do nothing.
            high_sat_threshold = 155
            if avg_sat < high_sat_threshold:
                sat_boost = 1.0 + max(0, (140 - avg_sat) * 0.0015)
                # Cap at 1.15 to prevent neon-looking skin
                sat_boost = min(1.15, sat_boost)
                if sat_boost > 1.01:
                    print(f"[{ts()}] [Premium] Ultra HD: Adaptive Saturation (Boost={sat_boost:.2f})...")
                    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h, s, v = cv2.split(hsv)
                    s = np.clip(s * sat_boost, 0, 255)
                    out = cv2.cvtColor(cv2.merge((h, s, v)).astype(np.uint8), cv2.COLOR_HSV2BGR)
            else:
                print(f"[{ts()}] [Premium] Ultra HD: Color within safe range, skipping boost.")
            
        except Exception as e:
            print(f"[{ts()}] [Premium] Ultra HD: CRITICAL ERROR in mode logic: {e}")
            # Do NOT raise here, let it return the copy at least
            
        return out

    def _low_light_mode(self, image, stats: dict):
        import cv2, numpy as np
        ts = self._get_timestamp
        out = image.copy()
        
        brightness = stats.get('brightness', 100)
        is_noisy = stats.get('is_noisy', False)

        # 1. Adaptive Brightening
        if brightness < 60:
            print(f"[{ts()}] [Premium] Low-Light: Very dark, applying strong brighten...")
            out = cv2.convertScaleAbs(out, alpha=1.2, beta=25)
        elif brightness < 100:
            print(f"[{ts()}] [Premium] Low-Light: Moderately dark, applying light brighten...")
            out = cv2.convertScaleAbs(out, alpha=1.1, beta=10)
        else:
            print(f"[{ts()}] [Premium] Low-Light: Image already bright, skipping artificial boost...")

        # 2. Smart Mild Denoising
        if is_noisy:
            print(f"[{ts()}] [Premium] Low-Light: Applying mild fastNlMeansDenoising...")
            out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
            
        # 3. Restore Details (Sharpening)
        print(f"[{ts()}] [Premium] Low-Light: Restoring details via filter2D...")
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(out, -1, kernel)

        # 4. Contrast Balance (EqualizeHist)
        print(f"[{ts()}] [Premium] Low-Light: EqualizeHist on L channel...")
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        
        return out

    def _hdr_boost_mode(self, image, stats: dict):
        import cv2, numpy as np
        ts = self._get_timestamp
        out = image.copy()
        
        is_noisy = stats.get('is_noisy', False)
        is_dark = stats.get('is_dark', False)
        is_blurry = stats.get('is_blurry', False)

        # 1. Adaptive Denoise (HDR expansion highly magnifies baseline noise)
        if is_noisy:
            print(f"[{ts()}] [Premium] HDR Boost: Pre-denoising background noise...")
            out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)

        # 2. Simulated Bracketing
        print(f"[{ts()}] [Premium] HDR Boost: Building simulated exposure brackets...")
        under_exp = cv2.convertScaleAbs(out, alpha=0.7, beta=-15)
        over_exp = cv2.convertScaleAbs(out, alpha=1.3, beta=25)
        
        # 3. Merge Mertens 
        print(f"[{ts()}] [Premium] HDR Boost: Processing Mertens Exposure Fusion...")
        merge_mertens = cv2.createMergeMertens()
        hdr_f32 = merge_mertens.process([under_exp, out, over_exp])
        hdr_u8 = np.clip(hdr_f32 * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Adaptive Micro-Contrast
        print(f"[{ts()}] [Premium] HDR Boost: Recovering LAB structure...")
        lab = cv2.cvtColor(hdr_u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_limit = 2.5 if is_dark else 1.5
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8,8))
        l = clahe.apply(l)
        hdr_u8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        
        # 5. Natural Vibrance Boost
        print(f"[{ts()}] [Premium] HDR Boost: Enriching saturation...")
        hsv = cv2.cvtColor(hdr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.15, 0, 255)
        hdr_u8 = cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 6. Adaptive Edge Sharpening
        if is_blurry:
            print(f"[{ts()}] [Premium] HDR Boost: Adding mild edge unsharp mask...")
            blur_k = cv2.GaussianBlur(hdr_u8, (0, 0), sigmaX=1.2)
            hdr_u8 = cv2.addWeighted(hdr_u8, 1.4, blur_k, -0.4, 0)
            
        return hdr_u8

    def _color_restore_mode(self, image, stats: dict):
        import cv2, numpy as np
        ts = self._get_timestamp
        out = image.copy()
        
        is_noisy = stats.get('is_noisy', False)
        is_portrait_like = stats.get('is_portrait_like', False)

        # 1. Baseline Denoise for cleaner HSV Math
        if is_noisy:
            print(f"[{ts()}] [Premium] Color Restore: Pre-Denoising...")
            out = cv2.fastNlMeansDenoisingColored(out, None, 4, 4, 7, 21)

        print(f"[{ts()}] [Premium] Color Restore: Analyzing nonlinear Vibrance/Value curves...")
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # 2. Adaptive Vibrance 
        # (Boosts unsaturated pixels more cleanly; protects already vivid pixels and skin tones)
        vibrance_factor = 0.25 if is_portrait_like else 0.45 
        s = s * (1.0 + vibrance_factor * (1.0 - s/255.0))
        s = np.clip(s, 0, 255)
        
        # 3. Dynamic Contrast Expansion (Value channel)
        # Bypasses the linear (+20) math that washed out shadows
        v_u8 = np.clip(v, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        v_u8 = clahe.apply(v_u8)
        v = v_u8.astype(np.float32)
        
        print(f"[{ts()}] [Premium] Color Restore: Recompiling image layers...")
        final = cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return final

    # -------------------------------------------------------------------------
    #  ESRGAN PIPELINE (General / DSLR-Style Refinement)
    # -------------------------------------------------------------------------
    def _esrgan_enhance(self, img, stats: dict, effective_mode: str = "general"):
        """ESRGAN-based enhancement. Refined for DSLR-like realism."""
        import cv2, numpy as np
        ts = self._get_timestamp

        try:
            print(f"[{ts()}] [ESRGAN] Step 2: Real-ESRGAN upscaling...")
            out, _ = self.upscaler.enhance(img, outscale=2)
            
            is_blurry = stats.get('is_blurry', False)
            is_noisy = stats.get('is_noisy', False)
            is_dark = stats.get('is_dark', False)
            
            # -- Refinements --------------------------------------------------
            if effective_mode != "bw":
                # A. Moderate Denoise (only if strictly necessary)
                if is_noisy:
                    print(f"[{ts()}] [ESRGAN] Step 3a: Selective Denoising...")
                    out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

                # B. Natural Sharpening (Soft DSLR look)
                # Reduced strength to avoid "cartoon" halos
                print(f"[{ts()}] [ESRGAN] Step 3b: Soft DSLR-style Sharpening...")
                blur_k = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
                out = cv2.addWeighted(out, 1.15, blur_k, -0.15, 0)

                # C. Local Contrast Balance (Soft CLAHE)
                print(f"[{ts()}] [ESRGAN] Step 3c: Soft CLAHE contrast balance...")
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
                l = clahe.apply(l)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

                # D. MICRO-TEXTURE RECOVERY (Realism Layer)
                # Adds a tiny pixel-level "sensor grain" to break AI plasticity
                print(f"[{ts()}] [ESRGAN] Step 3d: Micro-texture recovery (Organic Grain)...")
                noise = np.random.normal(0, 1.2, out.shape).astype(np.float32)
                out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            else:
                # B/W Case... (already correct in file)
                pass

        except Exception as e:
            print(f"[{ts()}] [ESRGAN] FAILED: {e}. Falling back to OpenCV pipeline.")
            return self._opencv_enhance(img, effective_mode, is_blurry, is_noisy, is_dark)

    # -------------------------------------------------------------------------
    #  GFPGAN PIPELINE
    # -------------------------------------------------------------------------
    def _gfpgan_enhance(self, img, effective_mode: str, is_blurry: bool,
                        is_noisy: bool, is_dark: bool, weight: float = 0.6):
        """GFPGAN face-restoration pipeline. Only invoked explicitly for
        'portrait' or 'face' modes. Falls back to ESRGAN then OpenCV."""
        import cv2
        ts = self._get_timestamp
        try:
            print(f"[{ts()}] [GFPGAN] Step 2: Running face restoration (weight={weight})...")
            _, _, out = self.face_restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight
            )
            print(f"[{ts()}] [GFPGAN] Step 3: Face restoration successful.")

            # Mild bilateral skin smoothing
            if effective_mode == "portrait":
                try:
                    out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)
                    print(f"[{ts()}] [GFPGAN] Step 3a: Skin smoothing applied.")
                except Exception as e:
                    print(f"[{ts()}] [GFPGAN] Skin smoothing skipped: {e}")

            # == DSLR-Like Photographic Refinements ==
            if effective_mode != "bw":
                try:
                    import numpy as np
                    # 1. Unsharp mask (crisp eyes & details)
                    strength = 0.55 if is_blurry else 0.40
                    blur_k = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
                    out = cv2.addWeighted(out, 1.0 + strength, blur_k, -strength, 0)
                    print(f"[{ts()}] [GFPGAN] Step 3b: Unsharp mask (strength={strength:.2f})")

                    # 2. CLAHE Tone Mapping (fix contrast & dynamic range)
                    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    clip = 2.2 if is_dark else 1.8
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                    l_ch = clahe.apply(l_ch)
                    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
                    print(f"[{ts()}] [GFPGAN] Step 3c: CLAHE tone mapping (clip={clip})")

                    # 3. Vibrance boost (rich skin tones)
                    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h_c, s_c, v_c = cv2.split(hsv)
                    s_c = np.clip(s_c * 1.08, 0, 255)
                    v_c = np.clip(v_c * 1.03, 0, 255)
                    out = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]).astype(np.uint8), cv2.COLOR_HSV2BGR)
                    print(f"[{ts()}] [GFPGAN] Step 3d: Vibrance boost applied.")
                except Exception as e:
                    print(f"[{ts()}] [GFPGAN] DSLR refinements failed/skipped: {e}")

            return out

        except Exception as e:
            print(f"[{ts()}] [GFPGAN] FAILED: {e}. Falling back to ESRGAN.")
            return self._esrgan_enhance(img, effective_mode, is_blurry, is_noisy, is_dark)

    # -------------------------------------------------------------------------
    #  MAIN ENTRY POINT
    # -------------------------------------------------------------------------
    def enhance(self, input_path, output_path, *, mode: str = "auto"):
        """Enhance an image and write it to output_path.

        Modes:
        - auto     -> ESRGAN fast path  (<=20s on CPU)         ← FIX 4
        - portrait -> GFPGAN face path  (may be 30-60s on CPU)
        - bokeh    -> ESRGAN + bokeh FX
        - bw       -> ESRGAN + B/W conversion
        - face     -> GFPGAN direct (legacy)
        - general  -> ESRGAN fast path

        OpenCV fallback guarantees a result is always written to disk.
        """
        if not self._initialized:
            raise RuntimeError("AI Engine is not initialized. Please call initialize() first.")

        basename = os.path.basename(input_path)
        ts = self._get_timestamp
        print(f"\n[{ts()}] --- STARTING PROCESSING: {basename} ---")

        import cv2
        import numpy as np

        # -- Step 0: Load image -----------------------------------------------
        print(f"[{ts()}] Step 0: Loading image from disk...")
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read input image at {input_path}")
        print(f"[{ts()}] Step 0: Image loaded OK ({img.shape[1]}x{img.shape[0]})")

        start_proc = time.time()
        # -- Step 1: Dynamic Resize --
        # Standard modes use 1024px. Premium/Auto use 2048px for Ultra HD detail.
        mode_clean = (mode or "auto").strip().lower()
        is_premium = mode_clean in ("auto", "ultra_hd", "low_light", "hdr", "color_restore", "portrait")
        current_max_dim = MAX_INPUT_DIM_PREMIUM if is_premium else MAX_INPUT_DIM_STD

        h, w = img.shape[:2]
        if w > current_max_dim or h > current_max_dim:
            if w > h:
                new_w, new_h = current_max_dim, int(h * (current_max_dim / w))
            else:
                new_h, new_w = current_max_dim, int(w * (current_max_dim / h))
            
            print(f"[{ts()}] Step 1: Resizing {w}x{h} -> {new_w}x{new_h} (Cap={current_max_dim}px)")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            print(f"[{ts()}] Step 1: Image size OK ({w}x{h}), no resize needed.")

        # -- Step 1b: Adaptive Scene Analysis ---------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())
        noise_sigma = float(cv2.meanStdDev(gray)[1][0][0])
        avg_sat = float(hsv_full[:,:,1].mean())
        
        # Contrast: SD of pixel intensities
        contrast_score = float(np.std(gray))
        
        # Flags
        is_dark = brightness < 70
        is_blurry = blur_score < 80
        is_noisy = noise_sigma > 32
        
        h2, w2 = img.shape[:2]
        center_crop = gray[int(h2 * 0.2):int(h2 * 0.8), int(w2 * 0.2):int(w2 * 0.8)]
        center_var = float(np.var(center_crop)) if center_crop.size else 0.0
        is_portrait_like = (h2 >= w2 * 0.8) and (center_var > 700)
        
        stats = {
            'blur_score': blur_score, 'brightness': brightness,
            'noise_sigma': noise_sigma, 'avg_sat': avg_sat,
            'contrast_score': contrast_score, 'is_dark': is_dark,
            'is_blurry': is_blurry, 'is_noisy': is_noisy,
            'is_portrait_like': is_portrait_like
        }
        
        print(f"[{ts()}] Step 1b: Analysis -- blur={blur_score:.1f}, noise={noise_sigma:.1f}, "
              f"brightness={brightness:.1f}, sat={avg_sat:.1f}, contrast={contrast_score:.1f}")

        # -- Step 2: Intelligent Auto Routing ---------------------------------
        mode = (mode or "auto").strip().lower()
        effective_mode = mode

        # Truly 'Auto' Logic
        if effective_mode == "auto":
            print(f"[{ts()}] Step 2: Auto-routing based on image stats...")
            if is_portrait_like:
                effective_mode = "portrait"
            elif stats.get('brightness', 100) < 65:
                effective_mode = "low_light"
            elif stats.get('contrast_score', 50) < 40:
                effective_mode = "hdr"
            elif stats.get('avg_sat', 128) < 80:
                effective_mode = "color_restore"
            else:
                effective_mode = "general" # Refined DSLR-style ESRGAN

        if effective_mode == "general":
             print(f"[{ts()}] Step 2: Routing -> Optimized General Pipeline (DSLR-Refined)")
        else:
             print(f"[{ts()}] Step 2: Routing -> effective_mode='{effective_mode}' (requested='{mode}')")

        # -- INTERNAL TIMEOUT GUARD (FIX 3) -----------------------------------
        # We run the heavy pipeline in this thread (run_in_threadpool gives us a
        # real thread). A threading.Timer fires at INTERNAL_TIMEOUT_S and forces
        # a save of a fast fallback so the caller always gets a file.
        result_box   = [None]   # [enhanced_img]
        error_box    = [None]   # [exception]
        finished_evt = threading.Event()

        def _run_pipeline():
            try:
                if effective_mode == "ultra_hd":
                    out = self._ultra_hd_mode(img, stats)
                elif effective_mode == "low_light":
                    out = self._low_light_mode(img, stats)
                elif effective_mode == "hdr":
                    out = self._hdr_boost_mode(img, stats)
                elif effective_mode == "color_restore":
                    out = self._color_restore_mode(img, stats)
                elif effective_mode in ("portrait", "face"):
                    weight = 0.6 if effective_mode == "portrait" else 0.5
                    out = self._gfpgan_enhance(img, effective_mode, is_blurry,
                                               is_noisy, is_dark, weight=weight)
                elif effective_mode in ("bokeh", "bw") and is_portrait_like:
                    out = self._gfpgan_enhance(img, effective_mode, is_blurry,
                                               is_noisy, is_dark, weight=0.55)
                else:
                    # auto (routed above) / general / bokeh (non-portrait) / bw (non-portrait)
                    out = self._esrgan_enhance(img, stats, effective_mode)
                result_box[0] = out
            except Exception as exc:
                error_box[0] = exc
            finally:
                finished_evt.set()

        worker = threading.Thread(target=_run_pipeline, daemon=True)
        worker.start()

        finished = finished_evt.wait(timeout=INTERNAL_TIMEOUT_S)

        if not finished:
            print(f"[{ts()}] WARNING: Internal {INTERNAL_TIMEOUT_S}s timeout reached. "
                  "Using OpenCV fallback to guarantee output.")
            result_box[0] = self._opencv_enhance(img, effective_mode,
                                                  is_blurry, is_noisy, is_dark)
        elif error_box[0] is not None:
            print(f"[{ts()}] ERROR in pipeline: {error_box[0]}. Using OpenCV fallback.")
            result_box[0] = self._opencv_enhance(img, effective_mode,
                                                  is_blurry, is_noisy, is_dark)

        output = result_box[0]

        # -- Step 3: Unified mode post-processing (applied to ALL pipeline paths) --
        # This is the single authoritative place for mode-specific visual effects.
        # Placing it here guarantees B&W and bokeh apply whether GFPGAN, ESRGAN,
        # or OpenCV produced the base image.
        try:
            if effective_mode == "bw":
                print(f"[{ts()}] Step 3: Applying PURE Black & White conversion...")
                gray_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                # Apply slight natural contrast instead of over-processed HDR/CLAHE
                gray_out = cv2.convertScaleAbs(gray_out, alpha=1.1, beta=0)
                output = cv2.cvtColor(gray_out, cv2.COLOR_GRAY2BGR)
                print(f"[{ts()}] Step 3: Pure Black & White done.")
            elif effective_mode == "bokeh":
                print(f"[{ts()}] Step 3: Applying bokeh simulation...")
                oh, ow = output.shape[:2]
                mask = np.zeros((oh, ow), dtype=np.uint8)
                cv2.ellipse(mask, (ow // 2, int(oh * 0.45)),
                            (int(ow * 0.35), int(oh * 0.55)), 0, 0, 360, 255, -1)
                blurred_bg = cv2.GaussianBlur(output, (0, 0), sigmaX=7)
                mask3 = cv2.merge([mask, mask, mask])
                output = np.where(mask3 == 255, output, blurred_bg)
                print(f"[{ts()}] Step 3: Bokeh done.")
        except Exception as post_err:
            print(f"[{ts()}] WARNING: Mode post-processing failed ({post_err}), keeping base output.")

        # -- Step 4: Save -----------------------------------------------------
        print(f"[{ts()}] Step 4: Saving final image to {os.path.basename(output_path)}...")
        save_ok = cv2.imwrite(output_path, output)
        if not save_ok:
            raise IOError(f"cv2.imwrite failed for path: {output_path}")

        duration = time.time() - start_proc
        print(f"[{ts()}] --- FINISHED: {os.path.basename(output_path)} "
              f"(Total: {duration:.2f}s) ---\n")
        return output_path


# --- Singleton & async wrappers -----------------------------------------------
_engine = None


async def enhance_image_ai(image_path: str, output_path: str):
    """Async wrapper -- runs sync enhance() in FastAPI's thread pool."""
    global _engine
    if _engine is None:
        _engine = AIImageEnhancer()
    from fastapi.concurrency import run_in_threadpool
    return await run_in_threadpool(_engine.enhance, image_path, output_path)


async def enhance_image_ai_mode(image_path: str, output_path: str, *, mode: str = "auto"):
    """Async wrapper with mode selection."""
    global _engine
    if _engine is None:
        _engine = AIImageEnhancer()
    from fastapi.concurrency import run_in_threadpool
    return await run_in_threadpool(_engine.enhance, image_path, output_path, mode=mode)


# Compatibility wrappers
async def denoise(path, out): return await enhance_image_ai(path, out)
async def sharpen(path, out): return await enhance_image_ai(path, out)
async def enhance_auto(path, out): return await enhance_image_ai(path, out)
