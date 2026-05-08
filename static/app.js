const fileInput = document.querySelector("#fileInput");
const previewImage = document.querySelector("#previewImage");
const emptyState = document.querySelector("#emptyState");
const detectionsBody = document.querySelector("#detections");
const historyBody = document.querySelector("#history");
const faceCount = document.querySelector("#faceCount");
const totalTests = document.querySelector("#totalTests");
const qualityStatus = document.querySelector("#qualityStatus");
const lastTest = document.querySelector("#lastTest");
const modeStatus = document.querySelector("#modeStatus");
const modelNote = document.querySelector("#modelNote");
const camera = document.querySelector("#camera");
const cameraButton = document.querySelector("#cameraButton");
const captureButton = document.querySelector("#captureButton");
const downloadButton = document.querySelector("#downloadButton");
const clearButton = document.querySelector("#clearButton");
const resetCountButton = document.querySelector("#resetCountButton");
const canvas = document.querySelector("#captureCanvas");
const imageNumber = document.querySelector("#imageNumber");
const bestResult = document.querySelector("#bestResult");
const featureSummary = document.querySelector("#featureSummary");

let stream = null;
let latestImage = "";
let history = JSON.parse(localStorage.getItem("ageDetectionHistory") || "[]");
let localTestCount = Number(localStorage.getItem("ageDetectionTestCount") || "0");

window.addEventListener("DOMContentLoaded", () => {
  totalTests.textContent = localTestCount;
  imageNumber.textContent = localTestCount;
  renderHistory();
  if (window.location.protocol === "file:") {
    modeStatus.textContent = "Server needed";
    modelNote.textContent = "Please run run_app.bat and open http://127.0.0.1:8000. Upload and camera cannot work from the HTML file directly.";
    cameraButton.disabled = true;
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (file) {
    await analyzeBlob(file);
    fileInput.value = "";
  }
});

cameraButton.addEventListener("click", async () => {
  if (stream) {
    stopCamera();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    modeStatus.textContent = "Camera blocked";
    modelNote.textContent = "Camera access needs http://127.0.0.1:8000 or HTTPS. It will not work if the page is opened directly from the folder.";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = stream;
    await camera.play();
    cameraButton.textContent = "Stop camera";
    captureButton.disabled = false;
    modeStatus.textContent = "Camera ready";
    modelNote.textContent = "Camera is active. Capture frame will test once and switch the camera off automatically.";
  } catch (error) {
    modeStatus.textContent = "Camera blocked";
    modelNote.textContent = cameraErrorMessage(error);
  }
});

captureButton.addEventListener("click", async () => {
  if (!stream) return;
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    if (blob) {
      await analyzeBlob(blob);
      stopCamera();
      modelNote.textContent += " Camera turned off after one test.";
    }
  }, "image/jpeg", 0.92);
});

downloadButton.addEventListener("click", () => {
  if (!latestImage) return;
  const link = document.createElement("a");
  link.href = latestImage;
  link.download = "age-detection-result.jpg";
  link.click();
});

clearButton.addEventListener("click", () => {
  latestImage = "";
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  emptyState.style.display = "grid";
  faceCount.textContent = "0";
  qualityStatus.textContent = "Waiting";
  lastTest.textContent = "None";
  imageNumber.textContent = localTestCount;
  bestResult.textContent = "Waiting for a test";
  featureSummary.textContent = "No image analyzed yet";
  downloadButton.disabled = true;
  detectionsBody.innerHTML = `<tr><td colspan="7">No detections yet.</td></tr>`;
  modelNote.textContent = "Cleared. Upload a new image or start the camera.";
});

resetCountButton.addEventListener("click", () => {
  localTestCount = 0;
  history = [];
  localStorage.setItem("ageDetectionTestCount", "0");
  localStorage.setItem("ageDetectionHistory", "[]");
  totalTests.textContent = "0";
  imageNumber.textContent = "0";
  renderHistory();
  modelNote.textContent = "Test count and history reset.";
});

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
  cameraButton.textContent = "Start camera";
  captureButton.disabled = true;
}

async function analyzeBlob(blob) {
  if (window.location.protocol === "file:") {
    modeStatus.textContent = "Server needed";
    modelNote.textContent = "Run run_app.bat first, then open http://127.0.0.1:8000. Upload cannot call the Python app from a file page.";
    return;
  }
  setBusy(true);
  const formData = new FormData();
  formData.append("image", blob, "capture.jpg");

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed");
    renderResult(data);
  } catch (error) {
    modeStatus.textContent = "Error";
    modelNote.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") {
    return "Camera permission was denied. Click the browser camera icon near the address bar and allow access, then press Start camera again.";
  }
  if (error?.name === "NotFoundError") {
    return "No camera was found on this device. You can still use Choose image.";
  }
  if (error?.name === "NotReadableError") {
    return "The camera is already being used by another app. Close that app and try again.";
  }
  return "Camera could not be opened. Make sure you are using http://127.0.0.1:8000 and allow camera access in the browser.";
}

function setBusy(isBusy) {
  modeStatus.textContent = isBusy ? "Analyzing" : "Ready";
  fileInput.disabled = isBusy;
  cameraButton.disabled = isBusy;
  captureButton.disabled = isBusy || !stream;
}

function renderResult(data) {
  latestImage = data.annotatedImage;
  previewImage.src = latestImage;
  previewImage.style.display = "block";
  emptyState.style.display = "none";
  downloadButton.disabled = false;

  localTestCount += 1;
  localStorage.setItem("ageDetectionTestCount", String(localTestCount));
  faceCount.textContent = data.faceCount;
  totalTests.textContent = localTestCount;
  imageNumber.textContent = localTestCount;
  qualityStatus.textContent = qualityText(data.quality);
  lastTest.textContent = data.testedAt;
  modeStatus.textContent = data.mode;
  modelNote.textContent = data.note;
  bestResult.textContent = classificationText(data);
  featureSummary.textContent = featureText(data);

  if (!data.detections.length) {
    detectionsBody.innerHTML = `<tr><td colspan="7">No faces detected. Try a clearer front-facing photo.</td></tr>`;
  } else {
    detectionsBody.innerHTML = data.detections.map((item) => renderDetectionRow(item, localTestCount)).join("");
  }

  addHistory(data);
}

function renderDetectionRow(item, testNumber) {
  return `
    <tr>
      <td>${testNumber}</td>
      <td>${item.id}</td>
      <td><strong>${item.ageGroup}</strong></td>
      <td>${item.ageBucket}</td>
      <td>${Math.round(item.confidence * 100)}%</td>
      <td>${item.method}</td>
      <td>${item.warning || ""}</td>
    </tr>
  `;
}

function classificationText(data) {
  if (!data.detections.length) return "No face detected";
  const first = data.detections[0];
  return `${first.ageGroup} (${first.ageBucket}) from the OpenCV DNN Caffe model`;
}

function featureText(data) {
  const parts = [];
  if (data.faceCount === 1) parts.push("single face");
  if (data.faceCount > 1) parts.push(`${data.faceCount} faces`);
  if (data.quality?.status) parts.push(data.quality.status.toLowerCase());
  if (data.quality?.issues?.length) parts.push(data.quality.issues.join(", ").toLowerCase());
  const warnings = data.detections.filter((item) => item.warning).length;
  if (warnings) parts.push(`${warnings} model notes`);
  return parts.length ? parts.join(" - ") : "ready";
}

function qualityText(quality) {
  if (!quality) return "Waiting";
  const issueText = quality.issues?.length ? ` - ${quality.issues.join(", ")}` : "";
  return `${quality.status}${issueText}`;
}

function addHistory(data) {
  const summary = data.detections.length
    ? data.detections.map((item) => `${item.ageGroup} ${item.ageBucket}`).join(", ")
    : "No face";
  history.unshift({
    test: localTestCount,
    time: data.testedAt,
    model: data.mode,
    faces: data.faceCount,
    quality: qualityText(data.quality),
    summary,
  });
  history = history.slice(0, 10);
  localStorage.setItem("ageDetectionHistory", JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  if (!history.length) {
    historyBody.innerHTML = `<tr><td colspan="6">No test history yet.</td></tr>`;
    return;
  }
  historyBody.innerHTML = history
    .map(
      (item) => `
        <tr>
          <td>${item.test}</td>
          <td>${item.time}</td>
          <td>${item.model}</td>
          <td>${item.faces}</td>
          <td>${item.quality}</td>
          <td>${item.summary}</td>
        </tr>
      `,
    )
    .join("");
}
