/**
 * Passport Photo Verification — Student Portal frontend logic.
 * Handles: mode switching (upload/camera), file selection, camera capture,
 * submission to api/verify.php, and live rendering of checklist results.
 */

(function () {
  const modeUploadBtn = document.getElementById('modeUploadBtn');
  const modeCameraBtn = document.getElementById('modeCameraBtn');
  const uploadPane = document.getElementById('uploadPane');
  const cameraPane = document.getElementById('cameraPane');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const uploadPreview = document.getElementById('uploadPreview');

  const cameraVideo = document.getElementById('cameraVideo');
  const cameraCaptured = document.getElementById('cameraCaptured');
  const startCameraBtn = document.getElementById('startCameraBtn');
  const snapBtn = document.getElementById('snapBtn');
  const retakeBtn = document.getElementById('retakeBtn');
  const captureCanvas = document.getElementById('captureCanvas');

  const verifyBtn = document.getElementById('verifyBtn');
  const verifyBtnLabel = document.getElementById('verifyBtnLabel');
  const resultBanner = document.getElementById('resultBanner');
  const criteriaList = document.getElementById('criteriaList');
  const serviceWarning = document.getElementById('serviceWarning');

  let currentMode = 'upload';
  let selectedBlob = null;
  let cameraStream = null;

  // ---------- Mode switching ----------
  function setMode(mode) {
    currentMode = mode;
    modeUploadBtn.classList.toggle('active', mode === 'upload');
    modeCameraBtn.classList.toggle('active', mode === 'camera');
    uploadPane.classList.toggle('hidden', mode !== 'upload');
    cameraPane.classList.toggle('hidden', mode !== 'camera');

    if (mode !== 'camera' && cameraStream) {
      stopCamera();
    }
    updateVerifyButtonState();
  }

  modeUploadBtn.addEventListener('click', () => setMode('upload'));
  modeCameraBtn.addEventListener('click', () => setMode('camera'));

  // ---------- Upload handling ----------
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
  });

  function handleFileSelect(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert('Please select a JPEG, PNG, or WEBP image.');
      return;
    }
    selectedBlob = file;
    const url = URL.createObjectURL(file);
    uploadPreview.src = url;
    uploadPreview.classList.remove('hidden');
    resetChecklist();
    updateVerifyButtonState();
  }

  // ---------- Camera handling ----------
  startCameraBtn.addEventListener('click', async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          // width: { ideal: 1280 }, 
          // height: { ideal: window.innerWidth < 500 ? 720 : 1280 },
          height: { ideal: window.innerWidth < 500 ? 900 : 1600 }
        }
      });
      cameraVideo.srcObject = cameraStream;
      cameraVideo.classList.remove('hidden');
      cameraCaptured.classList.add('hidden');
      startCameraBtn.classList.add('hidden');
      snapBtn.classList.remove('hidden');
      retakeBtn.classList.add('hidden');
    } catch (err) {
      alert('Could not access camera: ' + err.message);
    }
  });

  snapBtn.addEventListener('click', () => {
    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, vw, vh);

    captureCanvas.toBlob((blob) => {
      selectedBlob = blob;
      cameraCaptured.src = URL.createObjectURL(blob);
      cameraCaptured.classList.remove('hidden');
      cameraVideo.classList.add('hidden');
      snapBtn.classList.add('hidden');
      retakeBtn.classList.remove('hidden');
      stopCamera(false);
      resetChecklist();
      updateVerifyButtonState();
    }, 'image/jpeg', 0.95);
  });

  retakeBtn.addEventListener('click', async () => {
    selectedBlob = null;
    cameraCaptured.classList.add('hidden');
    updateVerifyButtonState();
    startCameraBtn.click();
  });

  function stopCamera(clearVideo = true) {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    if (clearVideo) {
      startCameraBtn.classList.remove('hidden');
      snapBtn.classList.add('hidden');
      retakeBtn.classList.add('hidden');
      cameraVideo.classList.add('hidden');
    }
  }

  // ---------- Verify button state ----------
  function updateVerifyButtonState() {
    verifyBtn.disabled = !selectedBlob;
  }

  // Cache default descriptions on load
  document.querySelectorAll('.criteria-item').forEach((el) => {
    const descEl = el.querySelector('.body span');
    if (descEl && !el.dataset.defaultDesc) {
      el.dataset.defaultDesc = descEl.textContent;
    }
  });

  // ---------- Checklist rendering ----------
  function resetChecklist() {
    resultBanner.innerHTML = '';
    document.querySelectorAll('.criteria-item').forEach((el) => {
      el.classList.remove('pass', 'fail');
      el.classList.add('pending');
      el.querySelector('.status-dot').textContent = '•';
      const descEl = el.querySelector('.body span');
      if (descEl && el.dataset.defaultDesc) {
        descEl.textContent = el.dataset.defaultDesc;
      }
    });
  }

  function renderResults(result) {
    const checks = result.results || {};
    document.querySelectorAll('.criteria-item').forEach((el) => {
      const key = el.dataset.key;
      const check = checks[key];
      const dot = el.querySelector('.status-dot');
      const descEl = el.querySelector('.body span');
      if (!check) {
        el.classList.remove('pending', 'pass', 'fail');
        return;
      }
      el.classList.remove('pending');
      if (check.passed) {
        el.classList.add('pass');
        el.classList.remove('fail');
        dot.textContent = '✓';
      } else {
        el.classList.add('fail');
        el.classList.remove('pass');
        dot.textContent = '✕';
      }
      if (key === 'white_background') {
        descEl.textContent = check.passed
          ? (check.message || 'White bg accepted.')
          : (check.message || 'White background not accepted, please try again.');
      } else if (check.message) {
        descEl.textContent = check.message;
      }
    });

    if (result.overall_passed) {
      resultBanner.innerHTML = `
        <div class="result-banner pass">
          <div class="icon">✅</div>
          <div>
            <h3>Photo Approved</h3>
            <p>Your photo meets all university requirements. You may now submit it officially.</p>
          </div>
        </div>`;
    } else {
      resultBanner.innerHTML = `
        <div class="result-banner fail">
          <div class="icon">⚠️</div>
          <div>
            <h3>Photo Did Not Pass</h3>
            <p>Please review the failed items below, fix your photo, and try again.</p>
          </div>
        </div>`;
    }
  }

  // ---------- Submission ----------
  verifyBtn.addEventListener('click', async () => {
    if (!selectedBlob) return;
    serviceWarning.classList.add('hidden');
    verifyBtn.disabled = true;
    verifyBtnLabel.innerHTML = '<span class="spinner"></span> Verifying…';
    resetChecklist();

    const formData = new FormData();
    const filename = currentMode === 'camera' ? 'camera-capture.jpg' : (selectedBlob.name || 'upload.jpg');
    formData.append('photo', selectedBlob, filename);
    formData.append('source', currentMode);
    formData.append('student_name', document.getElementById('studentName').value.trim());
    formData.append('student_id', document.getElementById('studentId').value.trim());

    try {
      const resp = await fetch('api/verify.php', { method: 'POST', body: formData });
      let data = null;
      try {
        data = await resp.json();
      } catch (jsonErr) {
        throw new Error('Verification server returned an invalid response format.');
      }

      if (!resp.ok) {
        serviceWarning.textContent = (data && data.error) ? data.error : `Verification failed (HTTP ${resp.status}). Please try again.`;
        serviceWarning.classList.remove('hidden');
      } else {
        renderResults(data);
      }
    } catch (err) {
      serviceWarning.textContent = err.message || 'Network error contacting the verification server. Please try again.';
      serviceWarning.classList.remove('hidden');
    } finally {
      verifyBtn.disabled = false;
      verifyBtnLabel.textContent = 'Verify Photo';
      updateVerifyButtonState();
    }
  });
})();
