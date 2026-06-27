/* =========================================
   1. GLOBAL VARIABLES & SETUP
   ========================================= */

// const API_BASE =
//     window.location.hostname === "localhost"
//         ? "http://localhost:5000"
//         : ""; // Flask backend

const API_BASE =
    ["localhost", "127.0.0.1"].includes(window.location.hostname)
        ? "http://127.0.0.1:5000"
        : "";

/* =========================================
   SECURE API WRAPPER
========================================= */

async function secureFetch(url, options = {}) {

    const controller = new AbortController();

    const timeout = setTimeout(() => {
        controller.abort();
    }, 60000); // 60 second timeout

    try {

        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            credentials: "same-origin",
            cache: "no-store"
        });

        clearTimeout(timeout);

        if (!response.ok) {

            let message = "Server Error";

            try {
                const err = await response.json();
                message = err.error || message;
            }
            catch {}

            throw new Error(message);
        }

        return response;

    }
    catch (err) {

        clearTimeout(timeout);

        if (err.name === "AbortError") {
            throw new Error("Request timed out.");
        }

        throw err;
    }

}

/* ===== COUGHIE VIDEO ENGINE (FINAL FIX) ===== */

/* ===== FINAL PERFECT LOOP ENGINE ===== */

let isCoughPlaying = false;
let inferenceRunning = false;
let chatbotBusy = false;

function playCoughieAnimation(type) {
    const video = document.getElementById("coughie-video");
    if (!video) return;

    // GREETING MODE (LOOP FOREVER)
    if (type === "greet") {

        if (isCoughPlaying) return; // don't interrupt cough

        video.src = "assets/video_animation/coughie_greeting.webm";
        video.loop = true;   // KEY FIX
        video.currentTime = 0;

        video.play().catch(() => {});
    }

    // COUGH MODE (PLAY ONCE)
    else if (type === "cough") {

        if (isCoughPlaying) return;

        isCoughPlaying = true;

        video.loop = false; // no loop
        video.src = "assets/video_animation/coughie_coughs.webm";
        video.currentTime = 0;

        video.play().catch(() => {});

        // AFTER COUGH → RETURN TO LOOP
        video.onended = () => {
            isCoughPlaying = false;

            setTimeout(() => {
                playCoughieAnimation("greet");
            }, 200); // small smooth delay
        };
    }
}

/* ===== INTRO VIDEO LOGIC (RUN ONCE) ===== */
function handleIntroVideo() {

    const intro = document.getElementById("intro-overlay");
    const video = document.getElementById("intro-video");

    if (!intro || !video) return;

    if (sessionStorage.getItem("introPlayed")) {
        intro.remove();
        document.body.style.overflow = "auto";
        return;
    }

    document.body.style.overflow = "hidden";

    let introClosed = false;

    function closeIntro() {

        if (introClosed) return;
        introClosed = true;

        sessionStorage.setItem(
            "introPlayed",
            "true"
        );

        intro.classList.add("intro-hide");

        setTimeout(() => {

            intro.remove();

            document.body.style.overflow = "auto";

        }, 1000);
    }

    video.play().catch(() => {});

    video.addEventListener(
        "ended",
        closeIntro,
        { once: true }
    );
}

const userKey = 'spectroUser';
const isLoggedIn = localStorage.getItem(userKey) !== null;

const LOGIN_ATTEMPTS_KEY = "spectro_login_attempts";

const MAX_LOGIN_ATTEMPTS = 5;

const LOGIN_LOCK_TIME = 5 * 60 * 1000; // 5 minutes

/* MOCK USER DATABASE */
const defaultUsers = [
    {
        email: "admin@spectro.com",
        pass: "admin123",
        name: "Dr. Admin",
        hashed: false
    },
    {
        email: "user@test.com",
        pass: "123456",
        name: "Dr. User",
        hashed: false
    }
];

function getUsersDB() {

    try {

        const stored =
            localStorage.getItem("spectro_users_db");

        if (!stored)
            return [...defaultUsers];

        const parsed = JSON.parse(stored);

        if (!Array.isArray(parsed))
            throw new Error();

        return parsed;

    } catch {

        localStorage.removeItem("spectro_users_db");

        return [...defaultUsers];

    }

}

/* =========================================
   INPUT VALIDATION
========================================= */

function isValidEmail(email) {

    const regex =
        /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    return regex.test(email);

}

function isStrongPassword(password) {

    return (
        password.length >= 8 &&
        /[A-Z]/.test(password) &&
        /[a-z]/.test(password) &&
        /\d/.test(password)
    );

}

function sanitizeName(name) {

    return name
        .replace(/[^a-zA-Z\s.'-]/g, "")
        .replace(/\s+/g, " ")
        .trim();

}

/* =========================================
   SESSION CONFIGURATION
========================================= */

const SESSION_KEY = "spectro_session";

const SESSION_DURATION =
    60 * 60 * 1000;   // 1 hour

function createSession(user) {

    const session = {

        name: user.name,

        email: user.email,

        loginTime: Date.now(),

        expiresAt:
            Date.now() +
            SESSION_DURATION,

        version: 1

    };

    localStorage.setItem(
        SESSION_KEY,
        JSON.stringify(session)
    );

}

function getSession() {

    const raw =
        localStorage.getItem(
            SESSION_KEY
        );

    if (!raw)
        return null;

    try {

        return JSON.parse(raw);

    }

    catch {

        return null;

    }

}

function isSessionValid() {

    const session =
        getSession();

    if (!session)
        return false;

    if (
        Date.now() >
        session.expiresAt
    ) {

        localStorage.removeItem(
            SESSION_KEY
        );

        return false;

    }

    return true;

}

function refreshSession() {

    const session =
        getSession();

    if (!session)
        return;

    session.expiresAt =
        Date.now() +
        SESSION_DURATION;

    localStorage.setItem(
        SESSION_KEY,
        JSON.stringify(session)
    );

}

function getLoginAttempts() {

    const stored = localStorage.getItem(LOGIN_ATTEMPTS_KEY);

    if (!stored) {

        return {
            count: 0,
            lockedUntil: 0
        };

    }

    return JSON.parse(stored);

}

function saveLoginAttempts(data) {

    localStorage.setItem(
        LOGIN_ATTEMPTS_KEY,
        JSON.stringify(data)
    );

}

function clearLoginAttempts() {

    localStorage.removeItem(
        LOGIN_ATTEMPTS_KEY
    );

}

/* =========================================
   PASSWORD HASHING
========================================= */

async function hashPassword(password) {

    const encoder = new TextEncoder();

    const data = encoder.encode(password);

    const hashBuffer =
        await crypto.subtle.digest(
            "SHA-256",
            data
        );

    return Array.from(
        new Uint8Array(hashBuffer)
    )
    .map(b => b.toString(16).padStart(2,"0"))
    .join("");
}

function saveUserToDB(user) {

    const users = getUsersDB();

    if (users.length >= 10) {
        throw new Error("Maximum demo users reached.");
    }

    users.push(user);

    localStorage.setItem(
        "spectro_users_db",
        JSON.stringify(users)
    );
}

// ===== EXISTING FUNCTIONS ABOVE =====

// ADD HERE (top-level function section)
function animateCounter(el, target) {
    let count = 0;
    const duration = 2000; // slower (2 seconds)
    const steps = 60;
    const increment = target / steps;

    const update = () => {
        count += increment;
        if (count < target) {
            el.innerText = Math.floor(count);
            requestAnimationFrame(update);
        } else {
            el.innerText = target;
        }
    };

    update();
}

let metricsStarted = false;

function handleMetricsOnScroll() {
    const section = document.querySelector('.metrics-section');
    if (!section || metricsStarted) return;

    const rect = section.getBoundingClientRect();

    if (rect.top < window.innerHeight - 100) {
        metricsStarted = true;

        document.querySelectorAll(".metric-box h3").forEach(el => {
            let value = el.innerText.replace(/[^0-9]/g, "");
            if (value) animateCounter(el, parseInt(value));
        });
    }
}

/* =========================================
   DOM READY
   ========================================= */

document.addEventListener('DOMContentLoaded', () => {

    handleIntroVideo();
    updateNavUI();
    highlightActivePage();
    protectRestrictedPages();
    initializeDeveloperCards();

    const input = document.getElementById("user-input");

    if (input) {
        input.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (window.location.pathname.includes('result.html')) {
        animateResultPage();
        loadResultData(); // NEW
    }

    if (window.location.pathname.includes('report.html')) {
        loadReportHistory(); // NEW
    }

    if (window.location.pathname.includes('explanatory.html')) {
        loadDynamicInsights();
    }

    if (window.location.pathname.includes('visual_aids.html')) {
        loadVisualFingerprint();
    }

    if (window.location.pathname.includes('what_changed.html')) {
        loadCounterfactualAnalysis();
    }

    initScrollReveal();

    showProcessingOverlay(type);

    window.addEventListener("scroll", handleMetricsOnScroll);
    handleMetricsOnScroll();

    const pdfBtn =
    document.getElementById(
        "download-report-btn"
    );

    if (
        pdfBtn &&
        !localStorage.getItem(
            "latestPredictionSession"
        )
    ) {

        pdfBtn.disabled = true;

        pdfBtn.title =
            "Run a screening first";
    }
});


document.addEventListener(
    "click",
    refreshSession
);

document.addEventListener(
    "keydown",
    refreshSession
);

/* =========================================
   2. NAVIGATION & AUTH LOGIC
   ========================================= */

function updateNavUI() {
    const authBtn = document.getElementById('auth-btn');
    const userProfile = document.getElementById('user-profile');
    const user = localStorage.getItem(userKey);

    // ===== USER STATE =====
    if (user) {
        if (authBtn) authBtn.style.display = "none";
        if (userProfile) {
            userProfile.style.display = "flex";
            document.getElementById('user-name').innerText = user;
        }

        // ===== UNLOCK NAV LINKS =====
        document.querySelectorAll('.nav-link.locked').forEach(link => {
            link.classList.remove('locked');

            const icon = link.querySelector('i');
            if (icon) icon.remove();

            const target = link.getAttribute('onclick')?.match(/'(.*?)'/);
            if (target) link.href = target[1];

            link.removeAttribute('onclick');
        });

        // ===== ENABLE HERO BUTTON =====
        const heroBtn = document.querySelector('.hero-cta');
        if (heroBtn) heroBtn.classList.remove('locked');

    } else {
        if (authBtn) authBtn.style.display = "inline-flex";
        if (userProfile) userProfile.style.display = "none";

        // ===== DISABLE HERO BUTTON =====
        const heroBtn = document.querySelector('.hero-cta');
        if (heroBtn) heroBtn.classList.add('locked');
    }
}

function checkAccess(targetUrl) {

    if (
        localStorage.getItem(userKey) &&
        isSessionValid()
    ) {

        refreshSession();

        document.body.style.opacity = "0";
        document.body.style.transition = "opacity 0.5s ease";

        setTimeout(() => {
            window.location.href = targetUrl;
        }, 500);

    } else {

        alert("🔒 Your session has expired.\nPlease login again.");

        logout(false);

    }

    return false;
}


/* =========================================
   AUTH FORM TOGGLING
   ========================================= */

function showSignup() {

    document
        .getElementById("login-form")
        .classList.add("hidden");

    document
        .getElementById("signup-form")
        .classList.remove("hidden");

    document
        .getElementById("form-title")
        .textContent = "Create Account";

    document
        .getElementById("form-subtitle")
        .textContent =
        "Create your SpectroCough account.";

}

function showLogin() {

    document
        .getElementById("signup-form")
        .classList.add("hidden");

    document
        .getElementById("login-form")
        .classList.remove("hidden");

    document
        .getElementById("form-title")
        .textContent = "Welcome Back";

    document
        .getElementById("form-subtitle")
        .textContent =
        "Enter your credentials to access the screening tool.";

}

async function handleLogin(event) {

    const attempts = getLoginAttempts();

    if (Date.now() < attempts.lockedUntil) {

        const remaining = Math.ceil(
            (attempts.lockedUntil - Date.now()) / 60000
        );

        alert(
            `Too many failed login attempts.\n\nPlease wait ${remaining} minute(s).`
        );

        return;

    }
    event.preventDefault();

    const email = sanitizeInput(document.getElementById('login-email').value);
    const pass = sanitizeInput(document.getElementById('login-pass').value);

    const users = getUsersDB();

    const hashedPass = await hashPassword(pass);

    let user =
        users.find(u =>
            u.email === email &&
            (
                (u.hashed && u.pass === hashedPass) ||
                (!u.hashed && u.pass === pass)
            )
        );

    if (user) {

        if (!user.hashed) {

            user.pass = hashedPass;

            user.hashed = true;

            localStorage.setItem(
                "spectro_users_db",
                JSON.stringify(users)
            );

        }

        createSession(user);
        clearLoginAttempts();
        localStorage.setItem(userKey, user.name);
        localStorage.setItem("spectroUserEmail", user.email);
        console.log("User object:", user);
        console.log("Stored email:", localStorage.getItem("spectroUserEmail"));

        alert(`✅ Welcome ${user.name}`);

        window.location.href = "index.html";

    }
    else {

        attempts.count++;

        if (attempts.count >= MAX_LOGIN_ATTEMPTS) {

            attempts.lockedUntil =
                Date.now() + LOGIN_LOCK_TIME;

            attempts.count = 0;

            saveLoginAttempts(attempts);

            alert(
                "Too many failed login attempts.\n\nLogin has been locked for 5 minutes."
            );

            return;

        }

        saveLoginAttempts(attempts);

        alert(
            `❌ Invalid Credentials\n\nRemaining Attempts: ${
                MAX_LOGIN_ATTEMPTS - attempts.count
            }`
        );

    }
}

async function handleSignup(event) {
    event.preventDefault();

    const name = sanitizeName(document.getElementById("signup-name").value);

    const email = document.getElementById('signup-email').value.trim().toLowerCase();

    if (email.length > 100) {

        alert("Email address is too long.");

        return;

    }

    const pass = document.getElementById('signup-pass').value;

    if (!isValidEmail(email)) {

        alert("Please enter a valid email address.");

        return;

    }

    if (!isStrongPassword(pass)) {

        alert(
            "Password must contain:\n\n" +
            "• At least 8 characters\n" +
            "• One uppercase letter\n" +
            "• One lowercase letter\n" +
            "• One number"
        );

        return;

    }

    if (name.length < 3) {

        alert(
            "Please enter your full name."
        );

        return;

    }

    if (getUsersDB().find(u => u.email.toLowerCase() === email.toLowerCase())) {
        alert("An account with this email already exists.");
        return;
    }

    const hashedPassword =
        await hashPassword(pass);

    saveUserToDB({

        name,

        email,

        pass: hashedPassword,

        hashed: true

    });
    alert("✅ Account Created Successfully!");

    showLogin();
}

function logout(showConfirm = true) {

    if (!showConfirm || confirm("Logout?")) {

        localStorage.removeItem(userKey);
        localStorage.removeItem("spectroUserEmail");
        localStorage.removeItem(SESSION_KEY);

        window.location.href = "index.html";

    }

}


/* =========================================
   3. SCREENING PAGE LOGIC
   ========================================= */

function triggerFileUpload(type) {
    document.getElementById(`file-${type}`).click();
}

document.getElementById('file-steth')
?.addEventListener('change', function () {

    if (!this.files || this.files.length === 0)
        return;

    updateFileStatus(
        'steth',
        this.files[0].name
    );
});

document.getElementById('file-mic')
?.addEventListener('change', function () {

    if (!this.files || this.files.length === 0)
        return;

    updateFileStatus(
        'mic',
        this.files[0].name
    );
});

function updateFileStatus(type, fileName) {

    const statusText =
        document.getElementById(
            `status-${type}`
        );

    const dropZone =
        document.getElementById(
            `drop-zone-${type}`
        );

    if (statusText) {

        statusText.innerText =
            `✅ Selected: ${fileName}`;

        statusText.style.color =
            'var(--success)';
    }

    if (dropZone) {

        dropZone.style.borderColor =
            'var(--success)';
    }
}

/* REAL BACKEND AUDIO PROCESSING */


async function injectAudio(type) {


    const fileInput = document.getElementById(`file-${type}`);

    if (!fileInput || fileInput.files.length === 0) {
        alert("⚠️ No Audio Selected");
        return;
    }

    showProcessingOverlay(type);

    const formData = new FormData();

    formData.append(
        "audio",
        fileInput.files[0]
    );

    // ----------------------------------------------------
    // Analysis type
    // ----------------------------------------------------

    formData.append(
        "analysis_type",
        type === "steth"
            ? "Stethoscope Analysis"
            : "Microphone Analysis"
    );

    // ----------------------------------------------------
    // NEW: Audio modality routing
    // ----------------------------------------------------

    formData.append(
        "audio_type",
        type === "steth"
            ? "stethoscope"
            : "microphone"
    );

    // ----------------------------------------------------
    // Logged-in user (for report history)
    // ----------------------------------------------------

    formData.append(
        "user_email",
        localStorage.getItem("spectroUserEmail") || ""
    );

    try {
        const response = await secureFetch(`${API_BASE}/infer`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok)
            throw new Error(data.error || "Inference failed");

        data.analysis_type =
            type === "steth"
                ? "Stethoscope Analysis"
                : "Microphone Analysis";

        localStorage.setItem(
            "latestPredictionSession",
            JSON.stringify(data)
        );

        hideProcessingOverlay();

        window.location.href="result.html";

    } catch (err) {
        alert("Error: " + err.message);
        hideProcessingOverlay();
    }
}


function showProcessingOverlay(type){

    const overlay=document.getElementById("processing-overlay");

    const title=document.getElementById("loading-title");

    const status=document.getElementById("loading-status");

    const icon = document.getElementById("loading-icon");

    const animation = document.getElementById("loading-animation");

    if(type==="steth"){

        title.innerText =
            "Analyzing Stethoscope Recording";

        icon.className =
            "fa-solid fa-stethoscope";

        animation.classList.remove("mic-mode");

    }
    else{

        title.innerText =
            "Analyzing Microphone Recording";

        icon.className =
            "fa-solid fa-microphone-lines";

        animation.classList.add("mic-mode");

    }


    overlay.classList.remove("hidden");
    overlay.style.display = "flex";

    document.body.style.overflow="hidden";

    document.body.style.pointerEvents="none";

    overlay.style.pointerEvents="all";

    const steps=[

        "Standardizing respiratory audio...",

        "Extracting acoustic biomarkers...",

        "Generating Mel-Spectrogram...",

        "Running Hybrid AI inference...",

        "Predicting Respiratory illness..."

    ];

    let i=0;

    status.innerText=steps[0];

    window.loadingInterval=setInterval(()=>{

        i=(i+1)%steps.length;

        status.innerText=steps[i];

    },1300);

}

function hideProcessingOverlay(){

    clearInterval(window.loadingInterval);

    document.body.style.pointerEvents="";

    document.body.style.overflow="";

    const overlay = document.getElementById("processing-overlay");

    overlay.classList.add("hidden");
    overlay.style.display = "none";

}

/* =========================================
   4. RESULT PAGE DATA (NEW)
   ========================================= */

function loadResultData() {

    const stored =
    localStorage.getItem(
        "latestPredictionSession"
    );

    if (!stored) {

        alert(
            "No prediction session found. Please run a screening first."
        );

        window.location.href =
            "screening.html";

        return;
    }
    if (!stored) return;

    const result = JSON.parse(stored);

    const icon =
    document.getElementById(
        "disease-icon"
    );

    if (icon) {

        const cls =
            result.predicted_class.toLowerCase();

        if (
            cls.includes("healthy")
        ) {

            icon.className =
                "fa-solid fa-heart-circle-check";

        } else {

            icon.className =
                "fa-solid fa-lungs-virus";
        }
    }

    const diseaseMap = {
        asthma: "ASTHMA",
        bronchial: "BRONCHITIS",
        copd: "COPD",
        pneumonia: "PNEUMONIA",
        healthy: "HEALTHY",

        covid19: "COVID-19",
        healthy_cough: "HEALTHY COUGH",
        sneezing: "NON-COUGH (SNEEZING)"
    };

    const formattedClass =
        diseaseMap[result.predicted_class.toLowerCase()]
        ||
        result.predicted_class
            .replaceAll("_", " ")
            .toUpperCase();

    document.getElementById(
        "predicted-disease"
    ).innerText =
    formattedClass;

    const percentage = (result.confidence * 100).toFixed(2) + "%";

    document.getElementById("prob-percentage").innerText = percentage;
    document.getElementById("prob-fill").style.width = percentage;

    const badge = document.getElementById("panel-badge");
    if (badge) badge.innerText = result.analysis_type || "AI Analysis";

    // NEW → probability breakdown
    const container = document.getElementById("probability-breakdown");

    if (container && result.probabilities) {

        container.innerHTML = "";

        const diseaseMap = {
            asthma: "ASTHMA",
            bronchial: "BRONCHITIS",
            copd: "COPD",
            pneumonia: "PNEUMONIA",
            healthy: "HEALTHY",

            covid19: "COVID-19",
            healthy_cough: "HEALTHY COUGH",
            sneezing: "NON-COUGH (SNEEZING)"
        };

        const sortedProbabilities =
            Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1]);

        sortedProbabilities.forEach(([cls, prob]) => {

            const percentage =
                (prob * 100).toFixed(2);

            const prettyName =
                diseaseMap[cls.toLowerCase()] ||
                cls.replaceAll("_", " ").toUpperCase();

            const isWinner =
                cls.toLowerCase() ===
                result.predicted_class.toLowerCase();

            container.innerHTML += `

                <div class="probability-card ${isWinner ? "winner-card" : ""}">

                    <div class="probability-header">

                        <span class="probability-name">
                            ${prettyName}
                        </span>

                        <span class="probability-value">
                            ${percentage}%
                        </span>

                    </div>

                    <div class="probability-bar-bg">

                        <div
                            class="probability-bar-fill"
                            style="width:${percentage}%">
                        </div>

                    </div>

                </div>

            `;
        });
    }
}


/* =========================================
   5. REPORT PAGE DATA (NEW)
   ========================================= */

async function loadReportHistory() {

    const tbody = document.getElementById("report-body");
    if (!tbody) return;

    try {
        const res = await secureFetch(

            `${API_BASE}/reports`,

            {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    email: localStorage.getItem(
                        "spectroUserEmail"
                    )

                })

            }

        );

        const reports = await res.json();

        tbody.innerHTML = "";

        reports.reverse().forEach((r, index) => {
            const row = `
                <tr>
                    <td>${index + 1}</td>
                    <td>${r.timestamp}</td>
                    <td>${localStorage.getItem(userKey) || "User"}</td>
                    <td><span class="status-badge">${r.predicted_class}</span></td>
                    <td>${(r.confidence * 100).toFixed(2)}%</td>
                </tr>
            `;
            tbody.innerHTML += row;
        });

    } 
    catch (err) {

        console.error(
            "Report fetch failed",
            err
        );

        tbody.innerHTML = `
            <tr>
                <td colspan="5"
                    style="
                    text-align:center;
                    padding:20px;">
                    Unable to load report history.
                </td>
            </tr>
        `;
    }
}


/* =========================================
   6. REPORT PAGE LOGIC
   ========================================= */

async function downloadPDF() {

    const { jsPDF } = window.jspdf;

    const sessionData =
        localStorage.getItem(
            "latestPredictionSession"
        );

    if (!sessionData) {
        alert("No report available.");
        return;
    }

    const data = JSON.parse(sessionData);

    const doc = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4"
    });

    const PAGE_WIDTH = 210;
    const PAGE_HEIGHT = 297;

    let y = 20;

    function addPageIfNeeded(extra = 15) {

        if (y + extra > PAGE_HEIGHT - 20) {

            doc.addPage();

            y = 20;
        }
    }

    function sectionTitle(title) {

        addPageIfNeeded(15);

        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");

        doc.setTextColor(
            14,
            165,
            233
        );

        doc.text(
            title,
            14,
            y
        );

        y += 8;

        doc.setDrawColor(
            14,
            165,
            233
        );

        doc.line(
            14,
            y - 3,
            195,
            y - 3
        );

        y += 3;
    }

    function normalText(text) {

        addPageIfNeeded(10);

        doc.setFontSize(10);
        doc.setFont(
            "helvetica",
            "normal"
        );

        doc.setTextColor(
            30,
            30,
            30
        );

        const wrapped =
            doc.splitTextToSize(
                text,
                180
            );

        doc.text(
            wrapped,
            14,
            y
        );

        y += wrapped.length * 5 + 4;
    }

    /* =======================================
       HEADER
    ======================================= */

    doc.setFillColor(
        14,
        165,
        233
    );

    doc.rect(
        0,
        0,
        PAGE_WIDTH,
        28,
        "F"
    );

    doc.setTextColor(
        255,
        255,
        255
    );

    doc.setFont(
        "helvetica",
        "bold"
    );

    doc.setFontSize(18);

    doc.text(
        "SpectroCough AI Report",
        14,
        17
    );

    doc.setFontSize(9);

    doc.text(
        "Respiratory Acoustic Screening Report",
        14,
        23
    );

    y = 40;

    /* =======================================
       PATIENT INFO
    ======================================= */

    sectionTitle("Patient Information");

    doc.setFontSize(10);
    doc.setTextColor(0,0,0);

    doc.text(
        `User : ${
            localStorage.getItem(
                "spectroUser"
            ) || "Guest"
        }`,
        14,
        y
    );

    y += 6;

    doc.text(
        `Generated : ${new Date().toLocaleString()}`,
        14,
        y
    );

    y += 10;

    /* =======================================
       PREDICTION SUMMARY
    ======================================= */

    sectionTitle("Prediction Summary");

    const confidence =
        Number(
            data.confidence || 0
        ) * 100;

    doc.setFillColor(
        245,
        248,
        255
    );

    doc.roundedRect(
        14,
        y,
        180,
        28,
        3,
        3,
        "F"
    );

    y += 8;

    doc.setFont(
        "helvetica",
        "bold"
    );

    doc.setFontSize(12);

    doc.text(
        `Predicted Class : ${
            data.predicted_class || "N/A"
        }`,
        18,
        y
    );

    y += 8;

    doc.setFont(
        "helvetica",
        "normal"
    );

    doc.text(
        `Confidence : ${confidence.toFixed(2)}%`,
        18,
        y
    );

    y += 8;

    doc.text(
        `Analysis Type : ${
            data.analysis_type ||
            "AI Analysis"
        }`,
        18,
        y
    );

    y += 15;

    /* =======================================
       CLASS PROBABILITIES
    ======================================= */

    if (data.probabilities) {

        sectionTitle(
            "Class Probability Distribution"
        );

        Object.entries(
            data.probabilities
        ).forEach(
            ([cls, prob]) => {

                addPageIfNeeded(10);

                const pct =
                    (
                        prob * 100
                    ).toFixed(2);

                doc.setFont(
                    "helvetica",
                    "normal"
                );

                doc.text(
                    cls,
                    14,
                    y
                );

                doc.text(
                    `${pct}%`,
                    185,
                    y,
                    { align: "right" }
                );

                doc.setDrawColor(
                    220,
                    220,
                    220
                );

                doc.rect(
                    14,
                    y + 2,
                    140,
                    4
                );

                doc.setFillColor(
                    14,
                    165,
                    233
                );

                doc.rect(
                    14,
                    y + 2,
                    140 *
                    (prob || 0),
                    4,
                    "F"
                );

                y += 10;
            }
        );
    }

    /* =======================================
       LAYMAN EXPLANATION
    ======================================= */

    if (
        data.explanation &&
        data.explanation.layman_summary
    ) {

        sectionTitle(
            "Layman Explanation"
        );

        normalText(
            data.explanation
                .layman_summary
        );
    }

    /* =======================================
       SCIENTIFIC EXPLANATION
    ======================================= */

    if (
        data.explanation &&
        data.explanation.scientific_summary
    ) {

        sectionTitle(
            "Scientific Analysis"
        );

        normalText(
            data.explanation
                .scientific_summary
        );
    }

    /* =======================================
       FEATURE DEVIATIONS
    ======================================= */

    if (
        data.explanation &&
        data.explanation
            .top_deviating_features
    ) {

        sectionTitle(
            "Key Acoustic Deviations"
        );

        data.explanation
            .top_deviating_features
            .forEach(feature => {

                normalText(
                    `${feature.feature} | Z-score: ${feature.z_score.toFixed(2)} | Difference: ${feature.percent_difference.toFixed(2)}%`
                );

            });
    }

    /* =======================================
       DISCLAIMER
    ======================================= */

    addPageIfNeeded(30);

    sectionTitle(
        "Important Disclaimer"
    );

    normalText(
        "SpectroCough is an AI-assisted respiratory acoustic screening system. The generated results are intended only for educational and pre-screening purposes. They are not a medical diagnosis and must not replace consultation with qualified healthcare professionals."
    );

    /* =======================================
       FOOTER
    ======================================= */

    const pages =
        doc.internal.getNumberOfPages();

    for (
        let i = 1;
        i <= pages;
        i++
    ) {

        doc.setPage(i);

        doc.setFontSize(8);

        doc.setTextColor(
            120,
            120,
            120
        );

        doc.text(
            `SpectroCough | Page ${i} of ${pages}`,
            PAGE_WIDTH / 2,
            PAGE_HEIGHT - 8,
            { align: "center" }
        );
    }

    doc.save(
        `SpectroCough_Report_${Date.now()}.pdf`
    );
}


/* =========================================
   CHATBOT OVERLAY LOGIC
   ========================================= */

function openChatOverlay() {
    const overlay = document.getElementById('coughie-overlay');
    if (!overlay) return;
    overlay.classList.remove('hidden');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    setTimeout(() => {
        playCoughieAnimation("greet");
    }, 400);
}

function closeChatOverlay() {
    const overlay = document.getElementById('coughie-overlay');
    if (!overlay) return;
    overlay.classList.remove('active');
    overlay.classList.add('hidden');
    document.body.style.overflow = 'auto';
}

function sendTemplate(msg) {

    const input =
        document.getElementById(
            "user-input"
        );

    if (!input) return;

    input.value = msg;

    sendMessage();
}

async function sendMessage() {

    playCoughieAnimation("cough");
    const input = document.getElementById('user-input');
    if (!input || input.value.trim() === "") return;

    const message = input.value;
    input.value = "";

    const chatArea = document.getElementById('chat-content-area');

    chatArea.innerHTML += `<div class="user-msg">${message}</div>`;

    const stored = localStorage.getItem("latestPredictionSession");

    try {
        const typing=document.createElement("div");

        typing.className="typing-message";

        typing.innerHTML=`
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        `;

        chatArea.appendChild(typing);

        chatArea.scrollTop=chatArea.scrollHeight;
        const res = await secureFetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                session: stored ? JSON.parse(stored) : null
            })
        });

        const data = await res.json();
        typing.remove();

        const botDiv = document.createElement("div");

        botDiv.className = "bot-msg";

        botDiv.textContent =
            data.reply || "No response";

        chatArea.appendChild(botDiv);

        chatArea.scrollTop = chatArea.scrollHeight;

    } catch (err) {
        typing.remove();
        chatArea.innerHTML += `<div class="bot-msg">Error connecting.</div>`;
    }
}


/* =========================================
   HELPER FUNCTIONS
   ========================================= */

function highlightActivePage() {
    const currentPath = window.location.pathname.split('/').pop();
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) link.classList.add('active');
    });
}

function protectRestrictedPages() {
    const path = window.location.pathname;
    if ((path.includes('screening.html') || path.includes('report.html')) && !isLoggedIn) {
        document.body.style.display = 'none';
        alert("Login required");
        window.location.href = 'login.html';
    }
}

function initializeDeveloperCards() {

    const cards = document.querySelectorAll(".dev-card");

    if (!cards.length) return;

    cards.forEach(card => {

        let isAnimating = false;
        let isHovering = false;

        card.addEventListener("mouseenter", () => {

            isHovering = true;

            if (isAnimating || card.classList.contains("opened"))
                return;

            isAnimating = true;

            card.classList.add("animating");

            setTimeout(() => {

                card.classList.remove("animating");

                isAnimating = false;

                // Mouse still on card?
                if (isHovering) {

                    card.classList.add("opened");

                }

            }, 2000);

        });

        card.addEventListener("mouseleave", () => {

            isHovering = false;

            // If animation already finished
            if (!isAnimating) {

                card.classList.remove("opened");

            }

        });

    });

}

function animateResultPage() {
    setTimeout(() => {
        document.querySelectorAll('.progress-fill').forEach(bar => {
            const width = bar.getAttribute('data-width');
            if (width) bar.style.width = width;
        });
    }, 500);
}

function loadDynamicInsights() {

    const stored =
    localStorage.getItem(
        "latestPredictionSession"
    );

    if (!stored) {

        alert(
            "No prediction session found. Please run a screening first."
        );

        window.location.href =
            "screening.html";

        return;
    }

    let session = null;

    try {

        session = JSON.parse(stored);

    }
    catch {

        localStorage.removeItem(
            "latestPredictionSession"
        );

        session = null;
    }

    const exp = session.explanation;
    if (!exp) return;

    const cond = document.getElementById("predicted-condition");
    if (cond && session.predicted_class){
        const formattedClass =session.predicted_class.replaceAll("_", " ").replace(/\b\w/g,c => c.toUpperCase());

        cond.innerText =
            formattedClass;
    }

    /* ================================
       LAYMAN EXPLANATION
    ================================ */

    const layman = document.getElementById("layman-text");
    if (layman) {

        layman.innerText =
            exp.layman_summary ||
            "Layman explanation is not available for this session.";
    }

    /* ================================
       SCIENTIFIC EXPLANATION
    ================================ */

    const scientific = document.getElementById("scientific-text");
    if (scientific) {

        scientific.innerText =
            exp.scientific_summary ||
            "Scientific interpretation is not available for this session.";
    }

    /* ================================
       AI INSIGHT
    ================================ */

    const insight = document.getElementById("dynamic-insight-text");
    if (insight) {

        const predicted = session.predicted_class;
        const confidence = (session.confidence * 100).toFixed(2);

        insight.innerText =
            `The AI model predicted ${predicted.toUpperCase()} with ` +
            `${confidence}% confidence based on distinguishing acoustic features.`;
    }

    /* ================================
       ACOUSTIC FEATURE VALUES
    ================================ */

    const features = exp.user_feature_values;
    if (!features) return;

    const mapping = {
        "rms": "val-rms",
        "zcr": "val-zcr",
        "spectral_centroid": "val-centroid",
        "spectral_bandwidth": "val-bandwidth",
        "spectral_rolloff": "val-rolloff"
    };

    // MFCC value (optional)
    if (features.mfcc_mean_1 !== undefined) {
        const mfccEl = document.getElementById("val-mfcc");
        if (mfccEl)
            mfccEl.innerText = Number(features.mfcc_mean_1).toFixed(4);
    }

    Object.entries(mapping).forEach(([feature, elementId]) => {

        const el = document.getElementById(elementId);

        if (el && features[feature] !== undefined) {
            el.innerText = Number(features[feature]).toFixed(4);
        }

    });

}

function sanitizeInput(str) {
    const div = document.createElement('div');
    div.innerText = str;
    return div.innerHTML.replace(/['";=]/g, "");
}

function initScrollReveal() {
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add('visible');
        });
    });
    document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
}

/* =========================================
   VISUAL AIDS PAGE DATA
========================================= */

function loadVisualFingerprint() {


    const stored = localStorage.getItem("latestPredictionSession");
    if (!stored) return;


    const session = JSON.parse(stored);
    const spec = session.spectrogram_analysis;
    if (!spec) {
        console.error("Spectrogram analysis missing from session");
        return;
    }

    // user spectrogram
    const userImg = document.getElementById("user-spectrogram");
    if (userImg)
        userImg.src = "data:image/png;base64," + spec.user_spectrogram;

    // predicted reference
    

    // healthy reference
    const healthyImg = document.getElementById("healthy-ref-spectrogram");
    if (healthyImg)
        if (
            spec.healthy_comparison &&
            spec.healthy_comparison.reference_image
        ) {

            healthyImg.src =
                "data:image/png;base64," +
                spec.healthy_comparison.reference_image;
        }

    // all class comparison
    const container = document.getElementById("all-class-comparisons");
    if (!container) return;

    container.innerHTML = "";

    Object.entries(spec.all_class_comparisons || {}).forEach(([cls, data]) => {

        container.innerHTML += `

            <div class="comparison-grid">

                <!-- USER -->
                <div class="glass-card visual-card warning-border">
                    <div class="card-header">
                        <span class="badge warning">Your Sample</span>
                        <h3>User Spectrogram</h3>
                    </div>

                    <div class="img-container">
                        <img class="spectrogram-img"
                             src="data:image/png;base64,${spec.user_spectrogram}" />
                    </div>
                </div>

                <!-- DISEASE -->
                <div class="glass-card visual-card">
                    <div class="card-header">
                        <span class="badge success">${cls.toUpperCase()}</span>
                        <h3>${cls} Cough Pattern</h3>
                    </div>

                    <div class="img-container">
                        <img class="spectrogram-img"
                             src="${data.reference_image ? `data:image/png;base64,${data.reference_image}`:''}" />
                    </div>
                </div>

            </div>

        `;
    });
}

/* =========================================
   WHAT-IF LOGIC PAGE DATA
========================================= */

function loadCounterfactualAnalysis() {

    const stored = localStorage.getItem("latestPredictionSession");
    if (!stored) return;

    const session = JSON.parse(stored);

    const cf = session.counterfactual;

    if (!cf) return;

    const container =
        document.getElementById("logic-container");

    container.innerHTML = "";

    const diseaseMap = {
        asthma: "ASTHMA",
        bronchial: "BRONCHITIS",
        copd: "COPD",
        pneumonia: "PNEUMONIA",
        healthy: "HEALTHY",

        covid19: "COVID-19",
        healthy_cough: "HEALTHY COUGH",
        sneezing: "NON-COUGH (SNEEZING)"
    };

    (cf.comparisons || []).forEach(comp => {

        const title =
            diseaseMap[
                comp.target_class.toLowerCase()
            ] ||
            comp.target_class.toUpperCase();

        let featureHTML = "";

        (comp.top_changes || []).forEach(change => {

            const arrow =
                change.direction === "increase"
                ? "▲"
                : "▼";

            featureHTML += `

                <div class="cf-feature-card">

                    <div class="cf-feature-name">

                        ${change.feature.toUpperCase()}

                    </div>

                    <div class="cf-feature-values">

                        <div>

                            <span>Current</span>

                            <strong>${change.current_value.toFixed(4)}</strong>

                        </div>

                        <div>

                            <span>Target</span>

                            <strong>${change.target_value.toFixed(4)}</strong>

                        </div>

                        <div>

                            <span>Change</span>

                            <strong>

                                ${arrow}
                                ${Math.abs(change.difference).toFixed(4)}

                            </strong>

                            <small>

                                (${change.percent_change.toFixed(2)}%)

                            </small>

                        </div>

                    </div>

                </div>

            `;

        });

        let formattedExplanation =
            (comp.llm_explanation || "")

                // remove markdown bold
                .replace(/\*\*/g, "")

                // preserve paragraphs
                .replace(/\n/g, "<br>")

                // format only numbered list items
                .replace(/(^|\s)(\d+)\.\s/g, "$1<br><br><strong>$2.</strong> ");

        container.innerHTML += `

            <div class="logic-section">

                <h2 class="logic-title">

                    If cough shifts toward
                    <span>${title}</span>

                </h2>

                <div class="glass-card logic-feature-panel">

                    ${featureHTML}

                </div>

                <div class="glass-card logic-explanation-panel">

                    <h3>

                        <i class="fa-solid fa-brain"></i>

                        AI Interpretation

                    </h3>

                    <div class="logic-description">

                        ${formattedExplanation}

                    </div>

                </div>

            </div>

        `;

    });

}