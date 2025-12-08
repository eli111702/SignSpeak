(async function () {
  const PREDICTION_HOLD_TIME = 2000; // 2.0 seconds to display the prediction
  
  const video = document.getElementById('video'); 
  const annotatedImage = document.getElementById('annotatedImage'); 
  const capture = document.getElementById('capture');
  const predLabel = document.getElementById('predLabel');
  const predConf = document.getElementById('predConf');
  const status = document.getElementById('status');
  const progressEl = document.getElementById('progress');
  const neededEl = document.getElementById('needed');

  let stream = null;
  let running = false;
  const CAPTURE_INTERVAL = 200; // ms

  // New state variables for holding the prediction
  let isPredictionHeld = false;
  let predictionTimeoutHandle = null;

  // client_id persisted per browser
  let client_id = localStorage.getItem('sli_client_id');
  if (!client_id) {
    client_id = crypto.randomUUID();
    localStorage.setItem('sli_client_id', client_id);
  }

  async function startCamera() {
    try {
      // Set fixed resolution for consistency with model training data
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      video.srcObject = stream;
      status.textContent = "Camera starting...";
      
      const streamReady = new Promise(resolve => {
          video.addEventListener('loadedmetadata', () => {
              // Set canvas size based on video dimensions
              capture.width = video.videoWidth || 640;
              capture.height = video.videoHeight || 480;
              
              // Set the image display size to match the capture/video size
              annotatedImage.width = capture.width;
              annotatedImage.height = capture.height;

              // Immediately hide the raw video feed
              video.style.display = 'none'; 
              // Show the image element which will receive the server-annotated frames
              annotatedImage.style.display = 'block'; 

              status.textContent = "Camera ready. Press Start.";
              resolve(true);
          }, { once: true });
      });

      await streamReady;

      return true; // Indicate success
    } catch (e) {
      status.textContent = "Camera error: " + e.message;
      return false; // Indicate failure
    }
  }

  function stopCamera() {
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
      status.textContent = "Camera stopped";
    }
  }

  // capture frame as base64 jpeg
  function captureFrame() {
    const ctx = capture.getContext('2d');
    // Draw the current video frame onto the hidden canvas
    ctx.drawImage(video, 0, 0, capture.width, capture.height);
    // lower quality to reduce payload size
    const dataUrl = capture.toDataURL('image/jpeg', 0.6);
    return dataUrl;
  }
  
  // This function sends a frame to the server only if we are not in a prediction hold state
  async function postFrameAndGetPrediction(dataUrl) {
    if (isPredictionHeld) {
        // We are holding the prediction display, so we don't send data to the server, 
        // but we keep looping to update the annotatedImage with the latest frame
        return; 
    }
    
    try {
      const MAX_RETRIES = 5;
      for (let i = 0; i < MAX_RETRIES; i++) {
        try {
          const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ client_id, model_key: MODEL_KEY, image: dataUrl })
          });
          
          if (!res.ok) {
            if (res.status >= 500) {
              throw new Error(`Server status ${res.status}`);
            }
          }
          
          const j = await res.json();

          // CRITICAL: Display the annotated image from the backend
          if (j.annotated_image) {
              annotatedImage.src = j.annotated_image;
          }

          if (j.status === 'collecting') {
            progressEl.textContent = j.progress;
            neededEl.textContent = j.needed || 30;
            predLabel.textContent = '...';
            predConf.textContent = '0.0';
            status.textContent = `Collecting sequence: ${j.progress}/${j.needed || 30} frames...`;
          } else if (j.status === 'predicted') {
            // New logic: Hold the prediction display
            isPredictionHeld = true;
            
            // Clear any existing timeout
            if (predictionTimeoutHandle) clearTimeout(predictionTimeoutHandle);

            predictionTimeoutHandle = setTimeout(() => {
                isPredictionHeld = false;
                predictionTimeoutHandle = null;
                // Optionally reset UI to 'collecting' state immediately after hold
                predLabel.textContent = '...';
                predConf.textContent = '0.0';
                progressEl.textContent = 0;
            }, PREDICTION_HOLD_TIME);

            // Update UI with prediction
            predLabel.textContent = j.label;
            predConf.textContent = (j.confidence * 100).toFixed(1) + "%";
            progressEl.textContent = j.progress;
            status.textContent = `PREDICTION: "${j.label}". New sequence starts in ${PREDICTION_HOLD_TIME/1000}s.`;
          
          } else if (j.status === 'no_hand') { 
            progressEl.textContent = j.progress;
            neededEl.textContent = j.needed || 30;
            predLabel.textContent = '...';
            predConf.textContent = '0.0';
            status.textContent = j.message; 
          } else if (j.status === 'error') {
            status.textContent = j.message || "Server error";
            stopLoop();
          }
          return;
        } catch (e) {
          if (i === MAX_RETRIES - 1) {
            throw e;
          }
          // Exponential backoff retry
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 500)); 
        }
      }

    } catch (e) {
      status.textContent = "Network error: " + e.message;
    }
  }

  async function clearServerSequence() {
    try {
        // Ensure prediction hold is released immediately
        if (predictionTimeoutHandle) clearTimeout(predictionTimeoutHandle);
        isPredictionHeld = false;

        await fetch('/stop_interpretation', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ client_id })
        });
        progressEl.textContent = 0;
        predLabel.textContent = '---';
        predConf.textContent = '0.0%';
    } catch (e) {
        console.error("Failed to clear server sequence:", e);
    }
  }

  // loop
  let loopHandle = null;
  function startLoop() {
    if (running) return;
    running = true;
    clearServerSequence(); // Clear buffer when starting/restarting
    loopHandle = setInterval(() => {
      // Check if video is loaded and ready before capturing
      if (video && video.readyState >= 2) { 
        const dataUrl = captureFrame(); // Always capture the frame
        // Always send the current frame back to the server for visual annotation
        // We bypass the dataUrl check for the prediction post, as the loop handle ensures the frame capture.
        postFrameAndGetPrediction(dataUrl); 
      }
    }, CAPTURE_INTERVAL);
    status.textContent = "Running and collecting frames...";
    
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    if (btnStart) btnStart.disabled = true;
    if (btnStop) btnStop.disabled = false;
  }

  function stopLoop() {
    if (!running) return;
    running = false;
    clearInterval(loopHandle);
    clearServerSequence(); // Clear buffer immediately on stop
    status.textContent = "Stopped. Sequence cleared.";
    
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    if (btnStart) btnStart.disabled = false;
    if (btnStop) btnStop.disabled = true;
  }

  // wire buttons if present
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  if (btnStart) {
      btnStart.onclick = startLoop;
      btnStart.disabled = true;
  }
  if (btnStop) {
      btnStop.onclick = stopLoop;
      btnStop.disabled = true;
  }

  // start camera on load and then ensure buttons are enabled if camera starts
  const cameraStarted = await startCamera();
  if (cameraStarted) {
    if (btnStart) btnStart.disabled = false; 
    status.textContent = "Ready to Interpret. Press Start.";
  }
})();