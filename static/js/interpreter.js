let running = false;
let client_id = "user_" + Math.random().toString(36).substring(2);
let intervalId = null;

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const stateReady = document.getElementById("state-ready");
const stateRunning = document.getElementById("state-running");
const stateStopped = document.getElementById("state-stopped");

const labelBox = document.getElementById("predicted-label");
const confBox = document.getElementById("predicted-confidence");


// -------------------- UI STATE SWITCH --------------------
function showState(state) {
    stateReady.classList.add("hidden");
    stateRunning.classList.add("hidden");
    stateStopped.classList.add("hidden");

    document.getElementById(state).classList.remove("hidden");
}


// -------------------- CAMERA --------------------
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

startCamera();


// -------------------- START INTERPRETATION --------------------
document.getElementById("btn-start").onclick = () => {
    running = true;
    showState("state-running");

    intervalId = setInterval(captureAndSend, 120); // ~8 FPS
};


// -------------------- STOP --------------------
document.getElementById("btn-stop").onclick = () => {
    running = false;
    clearInterval(intervalId);
    showState("state-stopped");
};


// -------------------- RESTART --------------------
document.getElementById("btn-restart").onclick = () => {
    running = false;
    clearInterval(intervalId);
    showState("state-ready");
};


// -------------------- CAPTURE → SEND TO BACKEND --------------------
async function captureAndSend() {
    if (!running) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let dataUrl = canvas.toDataURL("image/jpeg");

    const payload = {
        client_id: client_id,
        model_key: MODEL_KEY,
        image: dataUrl
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    const out = await res.json();

    if (out.status === "predicted") {
        labelBox.textContent = out.label;
        confBox.textContent = (out.confidence * 100).toFixed(1) + "%";
    }
}
